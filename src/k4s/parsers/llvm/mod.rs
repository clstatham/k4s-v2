use std::rc::Rc;

use llvm_ir::{
    types::{NamedStructDef, Typed, Types},
    ConstantRef, FPPredicate, Instruction, IntPredicate, Name, Operand, Type, TypeRef,
};

use crate::k4s::{
    contexts::llvm::FunctionContext, parsers::llvm::consteval::NameExt, Instr, InstrSize, Label,
    Linkage, Opcode, Register, Token,
};

use self::consteval::TypeExt;

pub mod consteval;

pub struct SsaInner {
    name: Name,
    ty: TypeRef,
    storage: Token,
    agg_const: Option<ConstantRef>,
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
    pub fn new(name: Name, ty: TypeRef, storage: Token, agg_const: Option<ConstantRef>) -> Self {
        Self {
            inner: SsaInner {
                name,
                ty,
                storage,
                agg_const,
            }
            .into(),
        }
    }

    pub fn parse_operand(op: &Operand, ctx: &mut FunctionContext, types: &Types) -> Self {
        match op {
            Operand::LocalOperand { name, ty } => ctx.get_or_push(name, ty, types),
            Operand::MetadataOperand => todo!("metadata operand"),
            Operand::ConstantOperand(con) => Self::parse_const(con, ctx.gen_name(), types),
        }
    }

    pub fn name(&self) -> &Name {
        &self.inner.name
    }

    pub fn ty(&self) -> TypeRef {
        self.inner.ty.to_owned()
    }

    pub fn agg_const(&self) -> Option<ConstantRef> {
        self.inner.agg_const.as_ref().cloned()
    }

    pub fn storage(&self) -> &Token {
        &self.inner.storage
    }

    pub fn instr_size(&self, types: &Types) -> InstrSize {
        self.ty().instr_size(types)
    }

    pub fn get_register(&self) -> Option<Register> {
        if let Token::Register(reg) = self.inner.storage {
            Some(reg)
        } else {
            None
        }
    }

    pub fn get_reg_offset(&self) -> Option<(i64, Register)> {
        if let Token::RegOffset(off, reg) = self.inner.storage {
            Some((off, reg))
        } else {
            None
        }
    }
    pub fn get_label_offset(&self) -> Option<(i64, Label)> {
        if let Token::LabelOffset(off, ref lab) = self.inner.storage {
            Some((off, lab.to_owned()))
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

    pub fn push_label(mut self, label: Name) -> Self {
        self.expr.push_label(label);
        self
    }
    pub fn push_comment(mut self, comment: &str) -> Self {
        self.expr.push_comment(comment);
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

pub enum ExprElem {
    Instr(Instr),
    Label(Name),
    Comment(String),
}

/// Like an `Instr`, but operates on `Ssa`'s instead of raw `Tokens`, and can contain multiple `Instrs`.
pub struct Expr {
    pub sequence: Vec<ExprElem>,
}

impl Expr {
    pub fn new() -> Self {
        Self {
            sequence: Vec::new(),
        }
    }

    pub fn builder() -> ExprBuilder {
        ExprBuilder::new()
    }

    pub fn push_instr(&mut self, instr: Instr) {
        self.sequence.push(ExprElem::Instr(instr));
    }
    pub fn push_label(&mut self, label: Name) {
        self.sequence.push(ExprElem::Label(label));
    }
    pub fn push_comment(&mut self, comment: &str) {
        self.sequence.push(ExprElem::Comment(comment.to_owned()));
    }

    pub fn parse(instr: &Instruction, ctx: &mut FunctionContext, types: &Types) -> Expr {
        macro_rules! arith_instr {
            ($instr:ident, $opcode:ident) => {{
                let a = Ssa::parse_operand(&$instr.operand0, ctx, types);
                let b = Ssa::parse_operand(&$instr.operand1, ctx, types);
                assert_eq!(a.ty().as_ref(), b.ty().as_ref());
                let dest = ctx.get_or_push(&$instr.dest, &a.ty(), types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(a.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::$opcode,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(b.storage().to_owned()),
                    ))
                    .build();

                expr
            }};
        }
        match instr {
            Instruction::Alloca(instr) => {
                let actual_name = ctx.gen_name();
                let actual = ctx.push(actual_name, instr.allocated_type.to_owned(), types);
                let ptr = ctx.push(
                    instr.dest.to_owned(),
                    Type::PointerType {
                        pointee_type: instr.allocated_type.to_owned(),
                        addr_space: 0,
                    }
                    .get_type(types),
                    types,
                );
                let (off, reg) = actual.get_reg_offset().unwrap();
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
                let dest =
                    ctx.get_or_push(&instr.dest, src.pointee_type().as_ref().unwrap(), types);
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

            Instruction::Add(instr) => arith_instr!(instr, Add),
            Instruction::Sub(instr) => arith_instr!(instr, Sub),
            Instruction::Mul(instr) => arith_instr!(instr, Mul),
            Instruction::UDiv(instr) => arith_instr!(instr, Div),
            Instruction::SDiv(instr) => arith_instr!(instr, Sdiv),
            Instruction::URem(instr) => arith_instr!(instr, Mod),
            Instruction::SRem(instr) => arith_instr!(instr, Smod),
            Instruction::Shl(instr) => arith_instr!(instr, Shl),
            Instruction::LShr(instr) => arith_instr!(instr, Shr),
            Instruction::AShr(instr) => arith_instr!(instr, Sshr),
            Instruction::Or(instr) => arith_instr!(instr, Or),
            Instruction::And(instr) => arith_instr!(instr, And),
            Instruction::Xor(instr) => arith_instr!(instr, Xor),
            Instruction::FAdd(instr) => arith_instr!(instr, Add),
            Instruction::FSub(instr) => arith_instr!(instr, Sub),
            Instruction::FMul(instr) => arith_instr!(instr, Mul),
            Instruction::FDiv(instr) => arith_instr!(instr, Div),

            Instruction::Call(instr) => {
                let func = if instr.function.is_right() {
                    let func = instr.function.as_ref().unwrap_right();
                    Ssa::parse_operand(func, ctx, types)
                } else {
                    let func_name = ctx.name().to_owned().strip_prefix();
                    let asm = instr.function.as_ref().unwrap_left();
                    let id = ctx.gen_name().strip_prefix();
                    let fn_name = format!("{}_inline_asm_{}", func_name, id);
                    let func = Ssa::new(
                        fn_name.to_owned().into(),
                        asm.get_type(types),
                        Token::Label(Label {
                            name: fn_name,
                            linkage: Linkage::NeedsLinking,
                        }),
                        None,
                    );
                    ctx.insert(func.clone());
                    func
                };
                let mut args = Vec::new();
                for (arg, _attrs) in instr.arguments.iter() {
                    if let Operand::MetadataOperand = arg {
                    } else {
                        args.push(Ssa::parse_operand(arg, ctx, types));
                    }
                }
                let dest = if let Type::FuncType { result_type, .. } = func.ty().as_ref() {
                    if let Type::VoidType = result_type.as_ref() {
                        None
                    } else {
                        instr
                            .dest
                            .as_ref()
                            .map(|dest| ctx.push(dest.to_owned(), result_type.to_owned(), types))
                    }
                } else if let Type::PointerType { pointee_type, .. } = func.ty().as_ref() {
                    if let Type::FuncType { result_type, .. } =
                        pointee_type.get_type(types).as_ref()
                    {
                        if let Type::VoidType = result_type.as_ref() {
                            None
                        } else {
                            instr.dest.as_ref().map(|dest| {
                                ctx.push(dest.to_owned(), result_type.to_owned(), types)
                            })
                        }
                    } else {
                        unreachable!("{:?}", func.ty().as_ref())
                    }
                } else {
                    unreachable!("{:?}", func.ty().as_ref())
                };

                let mut expr = Expr::new();

                for (arg, reg) in args.iter().zip(
                    [
                        Register::Rg,
                        Register::Rh,
                        Register::Ri,
                        Register::Rj,
                        Register::Rk,
                        Register::Rl,
                    ][..args.len().min(6)]
                        .iter()
                        .copied(),
                ) {
                    expr.push_instr(Instr::new(
                        Opcode::Mov,
                        arg.instr_size(types),
                        Some(Token::Register(reg)),
                        Some(arg.storage().to_owned()),
                    ))
                }
                // assert!(
                //     args.len() <= 6,
                //     "stack parameters are not currently supported"
                // );
                if args.len() > 6 {
                    for arg in args[6..].iter() {
                        expr.push_instr(Instr::new(
                            Opcode::Push,
                            arg.instr_size(types),
                            Some(arg.storage().to_owned()),
                            None,
                        ))
                    }
                }

                expr.push_instr(Instr::new(
                    Opcode::Call,
                    InstrSize::I64,
                    Some(func.storage().to_owned()),
                    None,
                ));
                if let Some(dest) = dest {
                    expr.push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(Token::Register(Register::Ra)),
                    ));
                }
                expr
            }
            Instruction::PtrToInt(instr) => {
                let ptr = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(ptr.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::IntToPtr(instr) => {
                let int = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(int.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::ZExt(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::Trunc(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::BitCast(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::SIToFP(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::FPToSI(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::UIToFP(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::FPToUI(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .build();
                expr
            }
            Instruction::SExt(instr) => {
                let src = Ssa::parse_operand(&instr.operand, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(src.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Sext,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        None,
                    ))
                    .build();
                expr
            }
            Instruction::ICmp(instr) => {
                let a = Ssa::parse_operand(&instr.operand0, ctx, types);
                let b = Ssa::parse_operand(&instr.operand1, ctx, types);
                let dest = ctx.get_or_push(
                    &instr.dest,
                    &Type::IntegerType { bits: 1 }.get_type(types),
                    types,
                );
                let jmp = match instr.predicate {
                    IntPredicate::EQ => Opcode::Jeq,
                    IntPredicate::NE => Opcode::Jne,
                    IntPredicate::SGT | IntPredicate::UGT => Opcode::Jgt,
                    IntPredicate::SLT | IntPredicate::ULT => Opcode::Jlt,
                    IntPredicate::SGE | IntPredicate::UGE => Opcode::Jge,
                    IntPredicate::SLE | IntPredicate::ULE => Opcode::Jle,
                };
                let cmp = match instr.predicate {
                    IntPredicate::EQ
                    | IntPredicate::NE
                    | IntPredicate::UGE
                    | IntPredicate::UGT
                    | IntPredicate::ULE
                    | IntPredicate::ULT => Opcode::Cmp,
                    IntPredicate::SGE
                    | IntPredicate::SGT
                    | IntPredicate::SLE
                    | IntPredicate::SLT => Opcode::Scmp,
                };
                let id = ctx.gen_name().strip_prefix();
                let true_lab = format!("{}_{}_{}_{}_true", ctx.name().strip_prefix(), cmp, jmp, id);
                let false_lab =
                    format!("{}_{}_{}_{}_false", ctx.name().strip_prefix(), cmp, jmp, id);
                let end_lab = format!("{}_{}_{}_{}_end", ctx.name().strip_prefix(), cmp, jmp, id);
                assert_eq!(a.ty().as_ref(), b.ty().as_ref());
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        cmp,
                        a.instr_size(types),
                        Some(a.storage().to_owned()),
                        Some(b.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: true_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: false_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(true_lab.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I8,
                        Some(dest.storage().to_owned()),
                        Some(Token::I8(1)),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(false_lab.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I8,
                        Some(dest.storage().to_owned()),
                        Some(Token::I8(0)),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(end_lab.into());
                expr.build()
            }

            Instruction::FCmp(instr) => {
                let a = Ssa::parse_operand(&instr.operand0, ctx, types);
                let b = Ssa::parse_operand(&instr.operand1, ctx, types);
                let dest = ctx.get_or_push(
                    &instr.dest,
                    &Type::IntegerType { bits: 1 }.get_type(types),
                    types,
                );
                let jmp = match instr.predicate {
                    FPPredicate::False => todo!(),
                    FPPredicate::True => todo!(),
                    FPPredicate::OEQ => Opcode::Jordeq,
                    FPPredicate::OGE => Opcode::Jordge,
                    FPPredicate::OGT => Opcode::Jordgt,
                    FPPredicate::OLE => Opcode::Jordle,
                    FPPredicate::OLT => Opcode::Jordlt,
                    FPPredicate::ONE => Opcode::Jordne,
                    FPPredicate::ORD => Opcode::Jord,
                    FPPredicate::UEQ => Opcode::Junoeq,
                    FPPredicate::UGE => Opcode::Junoge,
                    FPPredicate::UGT => Opcode::Junogt,
                    FPPredicate::ULE => Opcode::Junole,
                    FPPredicate::ULT => Opcode::Junolt,
                    FPPredicate::UNE => Opcode::Junone,
                    FPPredicate::UNO => Opcode::Juno,
                };
                let id = ctx.gen_name().strip_prefix();
                let true_lab = format!(
                    "{}_{}_{}_{}_true",
                    ctx.name().strip_prefix(),
                    Opcode::Cmp,
                    jmp,
                    id
                );
                let false_lab = format!(
                    "{}_{}_{}_{}_false",
                    ctx.name().strip_prefix(),
                    Opcode::Cmp,
                    jmp,
                    id
                );
                let end_lab = format!(
                    "{}_{}_{}_{}_end",
                    ctx.name().strip_prefix(),
                    Opcode::Cmp,
                    jmp,
                    id
                );
                assert_eq!(a.ty().as_ref(), b.ty().as_ref());
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Cmp,
                        a.instr_size(types),
                        Some(a.storage().to_owned()),
                        Some(b.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: true_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: false_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(true_lab.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I8,
                        Some(dest.storage().to_owned()),
                        Some(Token::I8(1)),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(false_lab.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        InstrSize::I8,
                        Some(dest.storage().to_owned()),
                        Some(Token::I8(0)),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_lab.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(end_lab.into());
                expr.build()
            }
            Instruction::GetElementPtr(instr) => {
                let addr = Ssa::parse_operand(&instr.address, ctx, types);

                let indices = instr
                    .indices
                    .iter()
                    .map(|idx| Ssa::parse_operand(idx, ctx, types))
                    .collect::<Vec<_>>();

                let (expr, _dest) = Self::gep(instr.dest.to_owned(), addr, indices, ctx, types);
                expr
            }
            Instruction::InsertValue(instr) => {
                let agg = Ssa::parse_operand(&instr.aggregate, ctx, types);
                let element = Ssa::parse_operand(&instr.element, ctx, types);
                let dest = ctx.get_or_push(&instr.dest, &agg.ty(), types);
                let mut current_ty = dest.ty().as_ref().to_owned();
                let mut offset: u64 = 0;
                for idx in instr.indices.iter() {
                    let idx = *idx as usize;
                    if let Type::NamedStructType { name } = current_ty {
                        if let NamedStructDef::Defined(ty) = types.named_struct_def(&name).unwrap()
                        {
                            current_ty = ty.as_ref().to_owned();
                        } else {
                            todo!("opaque structs")
                        }
                    }

                    match current_ty {
                        Type::StructType {
                            element_types,
                            is_packed: _,
                        } => {
                            offset += element_types[..idx]
                                .iter()
                                .map(|elem| elem.total_size_in_bytes(types) as u64)
                                .sum::<u64>();
                            current_ty = element_types[idx].as_ref().to_owned();
                        }
                        Type::ArrayType { element_type, .. }
                        | Type::VectorType { element_type, .. }
                        | Type::PointerType {
                            pointee_type: element_type,
                            ..
                        } => {
                            offset += element_type.total_size_in_bytes(types) as u64 * idx as u64;
                            current_ty = element_type.as_ref().to_owned();
                        }
                        _ => todo!("{:?}", current_ty),
                    }
                }

                let tmp = ctx.any_register().unwrap();
                let mut expr = Expr::new();
                let (off, reg) = dest.get_reg_offset().unwrap();
                expr.push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::Register(reg)),
                ));
                expr.push_instr(Instr::new(
                    Opcode::Sub,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::I64(-off as u64)),
                ));
                expr.push_instr(Instr::new(
                    Opcode::Add,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::I64(offset)),
                ));
                expr.push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Addr(Token::Register(tmp).into())),
                    Some(element.storage().to_owned()),
                ));

                ctx.take_back(tmp);
                expr
            }
            Instruction::ExtractValue(instr) => {
                let agg = Ssa::parse_operand(&instr.aggregate, ctx, types);
                let mut current_ty = agg.ty().as_ref().to_owned();
                let mut offset: u64 = 0;
                for idx in instr.indices.iter() {
                    let idx = *idx as usize;
                    if let Type::NamedStructType { name } = current_ty {
                        if let NamedStructDef::Defined(ty) = types.named_struct_def(&name).unwrap()
                        {
                            current_ty = ty.as_ref().to_owned();
                        } else {
                            todo!("opaque structs")
                        }
                    }

                    match current_ty {
                        Type::StructType {
                            element_types,
                            is_packed: _,
                        } => {
                            offset += element_types[..idx]
                                .iter()
                                .map(|elem| elem.total_size_in_bytes(types) as u64)
                                .sum::<u64>();
                            current_ty = element_types[idx].as_ref().to_owned();
                        }
                        Type::ArrayType { element_type, .. }
                        | Type::VectorType { element_type, .. }
                        | Type::PointerType {
                            pointee_type: element_type,
                            ..
                        } => {
                            offset += element_type.total_size_in_bytes(types) as u64 * idx as u64;
                            current_ty = element_type.as_ref().to_owned();
                        }
                        _ => todo!("{:?}", current_ty),
                    }
                }
                let dest = ctx.get_or_push(&instr.dest, &current_ty.get_type(types), types);
                let mut expr = Expr::new();
                let tmp = ctx.any_register().unwrap();
                let (off, reg) = agg.get_reg_offset().unwrap();
                expr.push_instr(Instr::new(
                    Opcode::Mov,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::Register(reg)),
                ));
                expr.push_instr(Instr::new(
                    Opcode::Sub,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::I64(-off as u64)),
                ));
                expr.push_instr(Instr::new(
                    Opcode::Add,
                    InstrSize::I64,
                    Some(Token::Register(tmp)),
                    Some(Token::I64(offset)),
                ));

                expr.push_instr(Instr::new(
                    Opcode::Mov,
                    dest.instr_size(types),
                    Some(dest.storage().to_owned()),
                    Some(Token::Addr(Token::Register(tmp).into())),
                ));

                ctx.take_back(tmp);

                expr
            }
            Instruction::Phi(instr) => {
                let last_block = ctx.last_block().to_owned();
                let dest = ctx.get_or_push(&instr.dest, &instr.to_type, types);

                let mut expr = Expr::new();
                let id = ctx.gen_name().strip_prefix();
                let end_lab = format!("{}_phi_{}_end", ctx.name().strip_prefix(), id);
                let mut ops = Vec::new();
                let mut labs = Vec::new();
                for (op, loc) in instr.incoming_values.iter() {
                    let id = ctx.gen_name().strip_prefix();
                    let lab = format!(
                        "{}_phi_{}_{}",
                        ctx.name().strip_prefix(),
                        id,
                        loc.strip_prefix()
                    );
                    let op = Ssa::parse_operand(op, ctx, types);
                    let loc = format!("{}_{}", ctx.name().strip_prefix(), loc.strip_prefix());
                    let loc = Token::Label(Label::new_unlinked(loc));
                    ops.push(op);
                    labs.push(lab.to_owned());

                    expr.push_instr(Instr::new(
                        Opcode::Cmp,
                        InstrSize::I64,
                        Some(last_block.to_owned()),
                        Some(loc),
                    ));
                    expr.push_instr(Instr::new(
                        Opcode::Jeq,
                        InstrSize::I64,
                        Some(Token::Label(Label::new_unlinked(lab))),
                        None,
                    ));
                }
                expr.push_instr(Instr::new(Opcode::Und, InstrSize::Unsized, None, None));
                for (op, lab) in ops.iter().zip(labs.iter()) {
                    expr.push_label(lab.to_owned().into());
                    expr.push_instr(Instr::new(
                        Opcode::Mov,
                        dest.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(op.storage().to_owned()),
                    ));
                    expr.push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label::new_unlinked(end_lab.to_owned()))),
                        None,
                    ));
                }
                expr.push_label(end_lab.into());

                expr
            }
            Instruction::Select(instr) => {
                let cond = Ssa::parse_operand(&instr.condition, ctx, types);
                let true_val = Ssa::parse_operand(&instr.true_value, ctx, types);
                let false_val = Ssa::parse_operand(&instr.false_value, ctx, types);
                assert_eq!(true_val.ty().as_ref(), false_val.ty().as_ref());
                let dest = ctx.get_or_push(&instr.dest, &true_val.ty(), types);
                let id = ctx.gen_name().strip_prefix();
                let true_dest = format!("{}_select_{}_true", ctx.name().strip_prefix(), id);
                let false_dest = format!("{}_select_{}_false", ctx.name().strip_prefix(), id);
                let end_dest = format!("{}_select_{}_end", ctx.name().strip_prefix(), id);
                let expr = Expr::builder()
                    .push_instr(Instr::new(
                        Opcode::Cmp,
                        cond.instr_size(types),
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
                    .push_label(true_dest.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        true_val.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(true_val.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_dest.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(false_dest.into())
                    .push_instr(Instr::new(
                        Opcode::Mov,
                        false_val.instr_size(types),
                        Some(dest.storage().to_owned()),
                        Some(false_val.storage().to_owned()),
                    ))
                    .push_instr(Instr::new(
                        Opcode::Jmp,
                        InstrSize::I64,
                        Some(Token::Label(Label {
                            name: end_dest.clone(),
                            linkage: Linkage::NeedsLinking,
                        })),
                        None,
                    ))
                    .push_label(end_dest.into())
                    .build();

                expr
            }
            Instruction::Freeze(_instr) => {
                Expr::new() // todo
            }

            _ => todo!("{:?}", instr),
        }
    }

    fn gep(
        dest: Name,
        addr: Ssa,
        indices: Vec<Ssa>,
        ctx: &mut FunctionContext,
        types: &Types,
    ) -> (Self, Ssa) {
        let mut expr = Expr::new();

        let tmp_dest = ctx.any_register().unwrap();

        if let Some((off, reg)) = addr.get_reg_offset() {
            expr.push_instr(Instr::new(
                Opcode::Mov,
                InstrSize::I64,
                Some(Token::Register(tmp_dest)),
                Some(Token::Register(reg)),
            ));
            expr.push_instr(Instr::new(
                Opcode::Sub,
                InstrSize::I64,
                Some(Token::Register(tmp_dest)),
                Some(Token::I64(-off as u64)),
            ));
        } else if let Some((off, lab)) = addr.get_label_offset() {
            expr.push_instr(Instr::new(
                Opcode::Mov,
                InstrSize::I64,
                Some(Token::Register(tmp_dest)),
                Some(Token::Label(lab)),
            ));
            expr.push_instr(Instr::new(
                Opcode::Add,
                InstrSize::I64,
                Some(Token::Register(tmp_dest)),
                Some(Token::I64(off as u64)),
            ));
        } else if let Token::Label(lab) = addr.storage() {
            expr.push_instr(Instr::new(
                Opcode::Mov,
                InstrSize::I64,
                Some(Token::Register(tmp_dest)),
                Some(Token::Label(lab.to_owned())),
            ));
        } else {
            unreachable!("Addr of GEP must be a pointer: {:?}", addr.storage())
        }

        // let mut current_type = Type::PointerType {
        //     pointee_type: addr.ty(),
        //     addr_space: 0,
        // };
        let mut current_type = addr.ty().as_ref().to_owned();

        for idx in indices.iter() {
            if let Type::NamedStructType { ref name } = current_type.clone() {
                let struc = types.named_struct_def(name).unwrap();
                if let NamedStructDef::Defined(ty) = struc {
                    current_type = ty.as_ref().to_owned();
                }
            }
            match current_type.clone() {
                Type::StructType {
                    ref element_types,
                    is_packed: _,
                } => {
                    let idx = idx.storage().as_integer::<u64>().unwrap() as usize;
                    let offset: usize = element_types[..idx]
                        .iter()
                        .map(|ty| ty.total_size_in_bytes(types))
                        .sum();
                    if offset > 0 {
                        expr.push_instr(Instr::new(
                            Opcode::Add,
                            InstrSize::I64,
                            Some(Token::Register(tmp_dest)),
                            Some(Token::I64(offset as u64)),
                        ));
                    }
                    current_type = element_types[idx].as_ref().to_owned();
                }
                Type::ArrayType { element_type, .. }
                | Type::VectorType { element_type, .. }
                | Type::PointerType {
                    pointee_type: element_type,
                    ..
                } => {
                    current_type = element_type.as_ref().to_owned();
                    let size = current_type.total_size_in_bytes(types);
                    if let Some(idx) = idx.storage().as_integer::<u64>() {
                        if idx > 0 {
                            expr.push_instr(Instr::new(
                                Opcode::Add,
                                InstrSize::I64,
                                Some(Token::Register(tmp_dest)),
                                Some(Token::I64((size as i64 * idx as i64) as u64)),
                            ));
                        }
                    } else {
                        let tmp = ctx.any_register().unwrap();
                        expr.push_instr(Instr::new(
                            Opcode::Mov,
                            InstrSize::I64,
                            Some(Token::Register(tmp)),
                            Some(idx.storage().to_owned()),
                        ));
                        expr.push_instr(Instr::new(
                            Opcode::Mul,
                            InstrSize::I64,
                            Some(Token::Register(tmp)),
                            Some(Token::I64(size as u64)),
                        ));
                        expr.push_instr(Instr::new(
                            Opcode::Add,
                            InstrSize::I64,
                            Some(Token::Register(tmp_dest)),
                            Some(Token::Register(tmp)),
                        ));
                        ctx.take_back(tmp);
                    }
                }
                ty => {
                    todo!("{:?}", ty)
                }
            }
        }

        let dest = ctx.get_or_push(
            &dest,
            &Type::PointerType {
                pointee_type: current_type.get_type(types),
                addr_space: 0,
            }
            .get_type(types),
            types,
        );
        expr.push_instr(Instr::new(
            Opcode::Mov,
            InstrSize::I64,
            Some(dest.storage().to_owned()),
            Some(Token::Register(tmp_dest)),
        ));
        ctx.take_back(tmp_dest);

        (expr, dest)
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self::new()
    }
}
