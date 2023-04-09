use std::rc::Rc;

use llvm_ir::{
    types::{Typed, Types},
    Instruction, Name, Operand, Type, TypeRef,
};

use crate::k4s::{
    contexts::llvm::{FunctionContext, LlvmContext},
    Instr, InstrSize, Opcode, Register, Token,
};

use self::consteval::TypeExt;

pub mod consteval;

pub struct SsaInner {
    name: Name,
    ty: TypeRef,
    storage: Token,
}

/// An extension of `Token` for use when parsing LLVM IR.
/// Due to the nature of Single Static Assignment form, SSA's are immutable once they're created.
/// This is enforced by having an `inner` component wrapped in an `Rc`, which also conveniently makes
/// SSA's reference counted and cheap to clone.
#[derive(Clone)]
pub struct Ssa {
    inner: Rc<SsaInner>,
}

impl Ssa {
    pub fn new(name: Name, ty: TypeRef, storage: Token) -> Self {
        Self {
            inner: SsaInner { name, ty, storage }.into(),
        }
    }

    pub fn parse_operand(op: &Operand, ctx: &mut FunctionContext, types: &Types) -> Self {
        if let Some(con) = op.as_constant() {
            return Self::parse_const(con, ctx.gen_name(), types);
        }
        match op {
            Operand::LocalOperand { name, ty } => ctx.get_or_push(name, ty),
            Operand::MetadataOperand => todo!("metadata operand"),
            Operand::ConstantOperand(_) => unreachable!(),
        }
    }

    pub fn name(&self) -> &Name {
        &self.inner.name
    }

    pub fn ty(&self) -> TypeRef {
        self.inner.ty.to_owned()
    }

    pub fn storage(&self) -> &Token {
        &self.inner.storage
    }

    pub fn instr_size(&self, types: &Types) -> InstrSize {
        InstrSize::from_integer_bits(self.ty().get_type(types).total_size_in_bytes() as u32 * 8)
            .unwrap()
    }

    pub fn get_register(&self) -> Option<Register> {
        if let Token::Register(reg) = self.inner.storage {
            Some(reg)
        } else {
            None
        }
    }

    pub fn get_offset(&self) -> Option<(i64, Register)> {
        if let Token::Offset(off, reg) = self.inner.storage {
            Some((off, reg))
        } else {
            None
        }
    }

    pub fn pointee_type(&self) -> Option<TypeRef> {
        if let Type::PointerType { pointee_type, .. } = self.inner.ty.as_ref() {
            Some(pointee_type.to_owned())
        } else {
            None
        }
    }
}

pub struct ExprBuilder {
    expr: Expr,
}

impl ExprBuilder {
    pub fn new() -> Self {
        Self { expr: Expr::new() }
    }

    pub fn push_instr(mut self, instr: Instr) -> Self {
        self.expr.push_instr(instr);
        self
    }

    pub fn build(self) -> Expr {
        self.expr
    }
}

impl Default for ExprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Like an `Instr`, but operates on `Ssa`'s instead of raw `Tokens`.
pub struct Expr {
    pub instrs: Vec<Instr>,
}

impl Expr {
    pub fn new() -> Self {
        Self { instrs: Vec::new() }
    }

    pub fn builder() -> ExprBuilder {
        ExprBuilder::new()
    }

    pub fn push_instr(&mut self, instr: Instr) {
        self.instrs.push(instr);
    }

    pub fn parse(instr: &Instruction, ctx: &mut FunctionContext, types: &Types) -> Expr {
        match instr {
            Instruction::Alloca(instr) => {
                let actual_name = ctx.gen_name();
                let actual = ctx.push(actual_name, instr.allocated_type.to_owned());
                let ptr = ctx.push(
                    instr.dest.to_owned(),
                    Type::PointerType {
                        pointee_type: instr.allocated_type.to_owned(),
                        addr_space: 0,
                    }
                    .get_type(types),
                );
                let (off, reg) = actual.get_offset().unwrap();
                let off = Token::I64(-off as u64);
                let reg = Token::Register(reg);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I64,
                        Some(ptr.storage().to_owned()),
                        Some(reg),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Sub,
                        InstrSize::I64,
                        Some(ptr.storage().to_owned()),
                        Some(off),
                    ))
                    .build();

                expr
            }
            Instruction::Load(instr) => {
                let src = Ssa::parse_operand(&instr.address, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, src.pointee_type().as_ref().unwrap());
                let tmp = ctx.any_register().unwrap();
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I64,
                        Some(Token::Register(tmp)),
                        Some(src.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(Token::Addr(Token::Register(tmp).into())),
                    ))
                    .build();
                ctx.take_back(tmp);
                expr
            }
            Instruction::Store(instr) => {
                let src = Ssa::parse_operand(&instr.value, ctx, types);
                let dest = Ssa::parse_operand(&instr.address, ctx, types);
                let tmp = ctx.any_register().unwrap();
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(Token::Register(tmp)),
                        Some(dest.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I64,
                        Some(Token::Addr(Token::Register(tmp).into())),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                ctx.take_back(tmp);
                expr
            }
            Instruction::Add(instr) => {
                let a = Ssa::parse_operand(&instr.operand0, ctx, types);
                let b = Ssa::parse_operand(&instr.operand1, ctx, types);
                assert_eq!(a.ty().as_ref(), b.ty().as_ref());
                let dest = ctx.get_or_push(&instr.dest, &a.ty());
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(a.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Add,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(b.storage().to_owned()),
                    ))
                    .build();

                expr
            }
            _ => todo!("{:?}", instr),
        }
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self::new()
    }
}
