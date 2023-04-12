use std::{
    fmt::Write,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::Result;
use llvm_ir::{types::Types, Constant, ConstantRef, Module, Name, Terminator, TypeRef};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::k4s::{
    parsers::llvm::{
        consteval::{NameExt, TypeExt},
        Expr, ExprElem, Ssa,
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
    last_block: Option<Token>,
}

impl FunctionContext {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub fn last_block(&self) -> &Token {
        self.last_block.as_ref().unwrap()
    }

    pub fn gen_name(&self) -> Name {
        Name::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    pub fn insert(&mut self, ssa: Ssa) {
        self.pool.insert(ssa.name().to_owned(), ssa);
    }

    pub fn get(&self, name: &Name) -> Option<Ssa> {
        self.pool.get(name).cloned()
    }

    pub fn get_or_push(&mut self, name: &Name, ty: &TypeRef, types: &Types) -> Ssa {
        if let Some(ssa) = self.pool.get(name) {
            ssa.clone()
        } else {
            self.push(name.to_owned(), ty.to_owned(), types)
        }
    }

    pub fn push(&mut self, name: Name, ty: TypeRef, types: &Types) -> Ssa {
        self.stack_offset -= ty.as_ref().total_size_in_bytes(types) as i64;
        self.stack_offset -= 8 - self.stack_offset.abs() % 8;
        let ssa = Ssa::new(
            name.to_owned(),
            ty,
            Token::RegOffset(self.stack_offset, Register::Bp),
            None,
        );
        self.pool.insert(name, ssa.clone());
        ssa
    }

    pub fn register(&mut self, reg: Register, ty: TypeRef) -> Option<Ssa> {
        if self.used_regs.insert(reg) {
            Some(Ssa::new(self.gen_name(), ty, Token::Register(reg), None))
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
                last_block: None,
            };
            let func_name: Name = func.name.to_owned().into();
            log::debug!("Entering function {}", func_name);
            self.functions.insert(func_name.to_owned(), ctx);
            let ctx = self
                .functions
                .get_mut(self.current_func.as_ref().unwrap())
                .unwrap();

            let last_block = Token::Register(ctx.any_register().unwrap());
            ctx.last_block = Some(last_block);

            let mut pop_args_expr = Expr::new();
            if func.parameters.len() > 6 {
                pop_args_expr.push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(Register::Rb)),
                    Some(Token::Register(Register::Sp)),
                ));
                pop_args_expr.push_instr(Instr::new(
                    Opcode::Add,
                    InstrSize::I64,
                    Some(Token::Register(Register::Sp)),
                    Some(Token::I64(8)),
                ));
                for param in func.parameters[6..].iter().rev() {
                    let ssa = ctx.push(param.name.to_owned(), param.ty.to_owned(), &types); // note that this doesn't actually push the stack
                    pop_args_expr.push_instr(Instr::new(
                        Opcode::Pop,
                        ssa.instr_size(&types),
                        Some(ssa.storage().to_owned()),
                        None,
                    ));
                }
                pop_args_expr.push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(Register::Sp)),
                    Some(Token::Register(Register::Rb)),
                ));
            }

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
                .push((func_name.to_owned(), vec![pop_args_expr, prologue]));

            let mut param_pushes = Vec::new();

            // assert!(
            //     func.parameters.len() <= 6,
            //     "stack parameters are not supported currently"
            // );

            for (param, reg) in func.parameters.iter().zip(
                [
                    Register::Rg,
                    Register::Rh,
                    Register::Ri,
                    Register::Rj,
                    Register::Rk,
                    Register::Rl,
                ][..func.parameters.len().min(6)]
                    .iter()
                    .copied(),
            ) {
                let ssa = ctx.push(param.name.to_owned(), param.ty.to_owned(), &types);
                let reg = ctx.register(reg, param.ty.to_owned()).unwrap();
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        ssa.instr_size(&types),
                        Some(ssa.storage().to_owned()),
                        Some(reg.storage().to_owned()),
                    ))
                    .build();
                param_pushes.push(expr);
            }
            // defer the argument pushes until we know the size of our stack frame
            // (will be added to prologue at end of phase 2)
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
                log::trace!("--> Entering basic block {}.", bb.name);
                let bb_name: Name =
                    format!("{}_{}", func_name.strip_prefix(), bb.name.strip_prefix()).into();
                for instr in bb.instrs.iter() {
                    log::trace!("-->     {}", instr);
                    exprs.push(Expr::builder().push_comment(&format!("{}", instr)).build());
                    exprs.push(Expr::parse(instr, ctx, &types));
                }

                exprs.push(
                    Expr::builder()
                        .push_instr(Instr::new(
                            Opcode::Mov,
                            InstrSize::I64,
                            Some(ctx.last_block.as_ref().unwrap().clone()),
                            Some(Token::Label(Label::new_unlinked(bb_name.strip_prefix()))),
                        ))
                        .build(),
                );

                log::trace!("-->     {}", bb.term);
                exprs.push(
                    Expr::builder()
                        .push_comment(&format!("{}", bb.term))
                        .build(),
                );

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
                    Terminator::CondBr(br) => {
                        let cond = Ssa::parse_operand(&br.condition, ctx, &types);
                        let true_dest = format!(
                            "{}_{}",
                            func_name.strip_prefix(),
                            br.true_dest.to_owned().strip_prefix()
                        );
                        let false_dest = format!(
                            "{}_{}",
                            func_name.strip_prefix(),
                            br.false_dest.to_owned().strip_prefix()
                        );
                        let expr = Expr::builder()
                            .push_instr(Instr::new(
                                Opcode::Cmp,
                                cond.instr_size(&types),
                                Some(cond.storage().to_owned()),
                                Some(Token::Register(Register::Rz)),
                            ))
                            .push_instr(Instr::new(
                                Opcode::Jne,
                                InstrSize::I64,
                                Some(Token::Label(Label {
                                    name: true_dest.clone(),
                                    linkage: Linkage::NeedsLinking,
                                })),
                                None,
                            ))
                            .push_instr(Instr::new(
                                Opcode::Jmp,
                                InstrSize::I64,
                                Some(Token::Label(Label {
                                    name: false_dest.clone(),
                                    linkage: Linkage::NeedsLinking,
                                })),
                                None,
                            ))
                            .push_instr(Instr::new(Opcode::Und, InstrSize::Unsized, None, None))
                            .build();
                        exprs.push(expr);
                    }
                    Terminator::Br(br) => {
                        let dest =
                            format!("{}_{}", func_name.strip_prefix(), br.dest.strip_prefix());
                        let expr = Expr::builder()
                            .push_instr(Instr::new(
                                Opcode::Jmp,
                                InstrSize::I64,
                                Some(Token::Label(Label::new_unlinked(dest))),
                                None,
                            ))
                            .build();
                        exprs.push(expr);
                    }
                    Terminator::Switch(switch) => {
                        let op = Ssa::parse_operand(&switch.operand, ctx, &types);
                        let mut expr = Expr::new();
                        for (case, dest) in switch.dests.iter() {
                            let case = Ssa::parse_const(case, ctx.gen_name(), &types);
                            let dest =
                                format!("{}_{}", func_name.strip_prefix(), dest.strip_prefix());

                            expr.push_instr(Instr::new(
                                Opcode::Cmp,
                                op.instr_size(&types),
                                Some(op.storage().to_owned()),
                                Some(case.storage().to_owned()),
                            ));
                            expr.push_instr(Instr::new(
                                Opcode::Jeq,
                                InstrSize::I64,
                                Some(Token::Label(Label::new_unlinked(dest))),
                                None,
                            ));
                        }
                        expr.push_instr(Instr::new(Opcode::Und, InstrSize::Unsized, None, None));
                        exprs.push(expr);
                    }
                    Terminator::Unreachable(_) => {
                        let expr = Expr::builder()
                            .push_instr(Instr::new(Opcode::Und, InstrSize::Unsized, None, None))
                            .build();
                        exprs.push(expr);
                    }
                    _ => todo!("{:?}", &bb.term),
                }

                ctx.body.push((bb_name, exprs));
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

        // globals
        for (global_name, global) in self.globals.iter() {
            if let Some(agg) = global.agg_const() {
                Self::dump_agg(&mut out, agg, global_name.to_owned(), &types);
            }
        }

        for (_func_name, func) in self.functions.iter() {
            for (block_name, block) in func.prologue.iter() {
                writeln!(out, "{}", Label::new_unlinked(block_name.strip_prefix()))?;
                for expr in block.iter() {
                    for elem in expr.sequence.iter() {
                        match elem {
                            ExprElem::Instr(instr) => writeln!(out, "    {}", instr)?,
                            ExprElem::Label(label) => {
                                writeln!(out, "{}", Label::new_unlinked(label.strip_prefix()))?
                            }
                            ExprElem::Comment(comment) => writeln!(out, ";{}", comment)?,
                        }
                    }
                }
            }
            for (block_name, block) in func.body.iter() {
                writeln!(out, "{}", Label::new_unlinked(block_name.strip_prefix()))?;
                for expr in block.iter() {
                    for elem in expr.sequence.iter() {
                        match elem {
                            ExprElem::Instr(instr) => writeln!(out, "    {}", instr)?,
                            ExprElem::Label(label) => {
                                writeln!(out, "{}", Label::new_unlinked(label.strip_prefix()))?
                            }
                            ExprElem::Comment(comment) => writeln!(out, ";{}", comment)?,
                        }
                    }
                }
            }
            for (block_name, block) in func.epilogue.iter() {
                writeln!(out, "{}", Label::new_unlinked(block_name.strip_prefix()))?;
                for expr in block.iter() {
                    for elem in expr.sequence.iter() {
                        match elem {
                            ExprElem::Instr(instr) => writeln!(out, "    {}", instr)?,
                            ExprElem::Label(label) => {
                                writeln!(out, "{}", Label::new_unlinked(label.strip_prefix()))?
                            }
                            ExprElem::Comment(comment) => writeln!(out, ";{}", comment)?,
                        }
                    }
                }
            }
            writeln!(out)?;
        }

        Ok(out)
    }

    fn dump_agg(out: &mut String, agg: ConstantRef, agg_name: Name, types: &Types) {
        if let Constant::Struct {
            name: _struc_name,
            values,
            is_packed: _,
        } = agg.as_ref()
        {
            // need to insert each element as a data tag with label pointers to their beginnings
            writeln!(out, "{}", Label::new_unlinked(agg_name.strip_prefix())).unwrap();
            for (i, elem) in values.iter().enumerate() {
                let elem_name = format!("{}_elem{}", agg_name.strip_prefix(), i);
                let elem = Ssa::parse_const(elem, elem_name.to_owned().into(), types);
                if let Some(agg2) = elem.agg_const() {
                    Self::dump_agg(out, agg2, elem_name.into(), types);
                } else if let Some(int) = elem.storage().as_integer::<u128>() {
                    let align = match elem.storage() {
                        Token::I8(_) => 1,
                        Token::I16(_) => 2,
                        Token::I32(_) => 4,
                        Token::I64(_) => 8,
                        Token::I128(_) => 16,
                        _ => unreachable!(),
                    };
                    writeln!(out, "@{} align{} ${}", elem_name, align, int).unwrap();
                } else {
                    match elem.storage() {
                        Token::Data(data) => {
                            write!(out, "@{} align1 \"", elem_name).unwrap();
                            for byte in data.data.iter() {
                                write!(out, "\\x{:02x}", *byte).unwrap();
                            }
                            writeln!(out, "\"").unwrap();
                        }
                        Token::LabelOffset(off, lab) => {
                            writeln!(out, "{} ({}+{})", Label::new_unlinked(elem_name), *off, lab)
                                .unwrap();
                        }
                        Token::Label(_lab) => {
                            writeln!(out, "{}", Label::new_unlinked(elem_name)).unwrap();
                        }
                        _ => todo!("{:?}", elem.storage()),
                    }
                }
            }
        } else {
            todo!("{:?}", agg.as_ref())
        }
    }
}
