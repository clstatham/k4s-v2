use std::{
    fmt::Write,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use llvm_ir::{
    types::{Typed, Types},
    Module, Name, Terminator, TypeRef,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::k4s::{
    parsers::llvm::{
        consteval::{NameExt, TypeExt},
        Expr, Ssa,
    },
    Instr, InstrSize, Label, Linkage, Opcode, Register, Token, GP_REGS,
};

pub struct FunctionContext {
    name: Name,
    next_id: AtomicUsize,
    prologue: Vec<(Name, Vec<Expr>)>,
    body: Vec<(Name, Vec<Expr>)>,
    epilogue: Vec<(Name, Vec<Expr>)>,
    pool: FxHashMap<Name, Ssa>,
    used_regs: FxHashSet<Register>,
    stack_offset: i64, // usually negative
}

impl FunctionContext {
    pub fn gen_name(&self) -> Name {
        Name::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    pub fn get(&self, name: &Name) -> Option<Ssa> {
        self.pool.get(name).cloned()
    }

    pub fn get_or_push(&mut self, name: &Name, ty: &TypeRef) -> Ssa {
        if let Some(ssa) = self.pool.get(name) {
            ssa.clone()
        } else {
            self.push(name.to_owned(), ty.to_owned())
        }
    }

    pub fn push(&mut self, name: Name, ty: TypeRef) -> Ssa {
        self.stack_offset -= ty.as_ref().total_size_in_bytes() as i64;
        self.stack_offset -= self.stack_offset.abs() % 8;
        let ssa = Ssa::new(
            name.to_owned(),
            ty,
            Token::Offset(self.stack_offset, Register::Bp),
        );
        self.pool.insert(name, ssa.clone());
        ssa
    }

    pub fn register(&mut self, reg: Register, ty: TypeRef) -> Option<Ssa> {
        if self.used_regs.insert(reg) {
            Some(Ssa::new(self.gen_name(), ty, Token::Register(reg)))
        } else {
            None
        }
    }

    pub fn any_register(&mut self) -> Option<Register> {
        GP_REGS
            .iter()
            .copied()
            .find(|&reg| self.used_regs.insert(reg))
    }

    pub fn take_back(&mut self, reg: Register) {
        self.used_regs.remove(&reg);
    }
}

pub struct LlvmContext {
    module: Module,
    globals: FxHashMap<Name, Ssa>,
    functions: FxHashMap<Name, FunctionContext>,
    current_func: Option<Name>,
}

impl LlvmContext {
    pub fn new(module: Module) -> Self {
        Self {
            module,
            globals: FxHashMap::default(),
            functions: FxHashMap::default(),
            current_func: None,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Self {
        Self::new(Module::from_bc_path(path).unwrap())
    }

    pub fn types(&self) -> &Types {
        &self.module.types
    }

    pub fn get_func(&self, name: &Name) -> Option<&FunctionContext> {
        self.functions.get(name)
    }

    pub fn current_func(&mut self) -> &mut FunctionContext {
        self.functions
            .get_mut(self.current_func.as_ref().unwrap())
            .unwrap()
    }

    pub fn lower(&mut self) -> Result<String> {
        let module = self.module.clone();
        let types = self.types().clone();

        // phase 1: parse globals
        for global in module.global_vars.iter() {
            let ssa = Ssa::parse_const(
                global
                    .initializer
                    .as_ref()
                    .expect("todo: non const globals"),
                global.name.to_owned(),
                &types,
            );
            self.globals.insert(global.name.to_owned(), ssa);
        }

        // phase 2: parse functions
        for func in module.functions.iter() {
            self.current_func = Some(func.name.to_owned().into());
            let ctx = FunctionContext {
                next_id: AtomicUsize::new(10000000),
                prologue: Vec::default(),
                body: Vec::default(),
                epilogue: Vec::default(),
                name: func.name.to_owned().into(),
                pool: self.globals.clone(),
                used_regs: FxHashSet::default(),
                stack_offset: 0,
            };
            let func_name: Name = func.name.to_owned().into();
            self.functions.insert(func_name.to_owned(), ctx);
            let ctx = self
                .functions
                .get_mut(self.current_func.as_ref().unwrap())
                .unwrap();

            let prologue = Expr::builder()
                .push_instr(Instr::new(
                    Opcode::Push,
                    InstrSize::I64,
                    Some(Token::Register(Register::Bp)),
                    None,
                ))
                .push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(Register::Bp)),
                    Some(Token::Register(Register::Sp)),
                ))
                .build();
            ctx.prologue
                .push((func.name.to_owned().into(), vec![prologue]));

            let mut param_pushes = Vec::new();

            assert!(
                func.parameters.len() <= 6,
                "stack parameters are not supported currently"
            );
            for (param, reg) in func.parameters.iter().zip(
                [
                    Register::Rg,
                    Register::Rh,
                    Register::Ri,
                    Register::Rj,
                    Register::Rk,
                    Register::Rl,
                ][..func.parameters.len()]
                    .iter()
                    .copied(),
            ) {
                let ssa = ctx.push(param.name.to_owned(), param.ty.to_owned());
                let reg = ctx.register(reg, param.ty.to_owned()).unwrap();
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        ssa.instr_size(&types),
                        Some(ssa.storage().to_owned()),
                        Some(reg.storage().to_owned()),
                    ))
                    .build();
                // defer the argument pushes until we know the size of our stack frame
                // (will be added to prologue at end of phase 2)
                param_pushes.push(expr);
            }
            ctx.body.push((
                format!("{}_push_params", ctx.name.to_owned().strip_prefix()).into(),
                param_pushes,
            ));

            let ret_label = format!("{}_ret", self.current_func.as_ref().unwrap().strip_prefix());
            let ret_label_tok = Token::Label(Label {
                name: ret_label.to_owned(),
                linkage: Linkage::NeedsLinking,
            });

            for bb in func.basic_blocks.iter() {
                let mut exprs = Vec::new();

                for instr in bb.instrs.iter() {
                    exprs.push(Expr::parse(instr, ctx, &types));
                }

                match &bb.term {
                    Terminator::Ret(ret) => {
                        let mut ret_expr = Expr::new();
                        if let Some(ret_op) = &ret.return_operand {
                            let ret_ssa = Ssa::parse_operand(ret_op, ctx, &types);
                            ret_expr.push_instr(Instr::new(
                                Opcode::Mov,
                                ret_ssa.instr_size(&types),
                                Some(Token::Register(Register::Ra)),
                                Some(ret_ssa.storage().to_owned()),
                            ))
                        }

                        ret_expr.push_instr(Instr::new(
                            Opcode::Jmp,
                            InstrSize::I64,
                            Some(ret_label_tok.to_owned()),
                            None,
                        ));
                        exprs.push(ret_expr);
                    }
                    _ => todo!("{:?}", &bb.term),
                }

                ctx.body.push((
                    format!("{}_{}", func_name.strip_prefix(), bb.name.strip_prefix()).into(),
                    exprs,
                ));
            }

            let epilogue = Expr::builder()
                .push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(Register::Sp)),
                    Some(Token::Register(Register::Bp)),
                ))
                .push_instr(Instr::new(
                    Opcode::Pop,
                    InstrSize::I64,
                    Some(Token::Register(Register::Bp)),
                    None,
                ))
                .push_instr(Instr::new(Opcode::Ret, InstrSize::Unsized, None, None))
                .build();
            let prologue_push_stack = Expr::builder()
                .push_instr(Instr::new(
                    Opcode::Sub,
                    InstrSize::I64,
                    Some(Token::Register(Register::Sp)),
                    Some(Token::I64(-ctx.stack_offset as u64)),
                ))
                .build();
            ctx.prologue.push((
                format!("{}_push_stack", func_name.strip_prefix()).into(),
                vec![prologue_push_stack],
            ));
            ctx.epilogue.push((ret_label.into(), vec![epilogue]));
        }

        // phase 3: translate to k4sm assembly
        let mut out = String::new();
        for (_func_name, func) in self.functions.iter() {
            for (block_name, block) in func.prologue.iter() {
                writeln!(out, "%{}", block_name.strip_prefix())?;
                for expr in block.iter() {
                    for instr in expr.instrs.iter() {
                        writeln!(out, "    {}", instr)?;
                    }
                }
            }
            for (block_name, block) in func.body.iter() {
                writeln!(out, "%{}", block_name.strip_prefix())?;
                for expr in block.iter() {
                    for instr in expr.instrs.iter() {
                        writeln!(out, "    {}", instr)?;
                    }
                }
            }
            for (block_name, block) in func.epilogue.iter() {
                writeln!(out, "%{}", block_name.strip_prefix())?;
                for expr in block.iter() {
                    for instr in expr.instrs.iter() {
                        writeln!(out, "    {}", instr)?;
                    }
                }
            }
            writeln!(out)?;
        }

        Ok(out)
    }
}
