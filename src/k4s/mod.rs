use std::{
    fmt::Display,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub},
};

pub mod parsers;
pub mod contexts;

pub trait Primitive
where
    Self: Sized + Copy,
{
    const MACHINE_CODE: u8;
    const ASM: &'static str;
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
    fn to_bytes(self) -> Box<[u8]>;
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct Unsized;

impl Primitive for Unsized {
    const ASM: &'static str = "";
    const MACHINE_CODE: u8 = 0;
    fn from_bytes(_: &[u8]) -> Option<Self> {
        None
    }

    fn to_bytes(self) -> Box<[u8]> {
        Box::new([])
    }
}

impl Primitive for u8 {
    const MACHINE_CODE: u8 = 1 << 0;
    const ASM: &'static str = "i8";

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 1 {
            Some(bytes[0])
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new([self])
    }
}
impl Primitive for u16 {
    const MACHINE_CODE: u8 = 1 << 1;
    const ASM: &'static str = "i16";
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 2 {
            Some(Self::from_le_bytes([bytes[0], bytes[1]]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}
impl Primitive for u32 {
    const MACHINE_CODE: u8 = 1 << 2;
    const ASM: &'static str = "i32";
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 4 {
            Some(Self::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}
impl Primitive for u64 {
    const MACHINE_CODE: u8 = 1 << 3;
    const ASM: &'static str = "i64";
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 8 {
            Some(Self::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}
impl Primitive for u128 {
    const MACHINE_CODE: u8 = 1 << 4;
    const ASM: &'static str = "i128";
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 16 {
            Some(Self::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}

impl Primitive for f32 {
    const MACHINE_CODE: u8 = 1 << 5;
    const ASM: &'static str = "f32";
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 4 {
            Some(Self::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3],
            ]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}
impl Primitive for f64 {
    const MACHINE_CODE: u8 = 1 << 6;
    const ASM: &'static str = "f64";
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 8 {
            Some(Self::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))
        } else {
            None
        }
    }
    fn to_bytes(self) -> Box<[u8]> {
        Box::new(self.to_le_bytes())
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Prim<T: Primitive>(pub T);

impl<T: Primitive + Add<T, Output = T>> Add<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn add(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T: Primitive + Sub<T, Output = T>> Sub<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn sub(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T: Primitive + Mul<T, Output = T>> Mul<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn mul(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl<T: Primitive + Div<T, Output = T>> Div<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn div(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl<T: Primitive + Rem<T, Output = T>> Rem<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn rem(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl<T: Primitive + Shr<T, Output = T>> Shr<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn shr(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 >> rhs.0)
    }
}

impl<T: Primitive + Shl<T, Output = T>> Shl<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn shl(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 << rhs.0)
    }
}

impl<T: Primitive + BitAnd<T, Output = T>> BitAnd<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn bitand(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl<T: Primitive + BitOr<T, Output = T>> BitOr<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn bitor(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl<T: Primitive + BitXor<T, Output = T>> BitXor<Prim<T>> for Prim<T> {
    type Output = Prim<T>;
    fn bitxor(self, rhs: Prim<T>) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}


#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum Linkage {
    Linked(u64), // addr
    NeedsLinking,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Label {
    pub name: String,
    pub linkage: Linkage,
}

#[derive(Debug, Clone)]
pub struct Data {
    pub label: Label,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone)]
pub enum Token {
    Unknown,
    I8(u8),
    I16(u16),
    I32(u32),
    I64(u64),
    I128(u128),
    F32(f32),
    F64(f64),
    Addr(Box<Token>),
    Offset(i64, Register),
    Register(Register),
    Label(Label),
    Data(Data),
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "(UNKNOWN)"),
            Self::I8(v) => write!(f, "{}", v),
            Self::I16(v) => write!(f, "{}", v),
            Self::I32(v) => write!(f, "{}", v),
            Self::I64(v) => write!(f, "{}", v),
            Self::I128(v) => write!(f, "{}", v),
            Self::F32(v) => write!(f, "{}", v),
            Self::F64(v) => write!(f, "{}", v),
            Self::Addr(v) => write!(f, "[{}]", v),
            Self::Offset(off, reg) => write!(f, "[{}+{}]", *off as i64, reg),
            Self::Register(reg) => write!(f, "{}", reg),
            Self::Label(lab) => write!(f, "%{}", lab.name),
            Self::Data(dat) => write!(f, "@{}", dat.label.name),
        }
    }
}

impl Token {
    pub const fn mc_size_in_bytes(&self) -> usize {
        match self {
            Self::I8(_) => 1 + 1,
            Self::I16(_) => 2 + 1,
            Self::I32(_) => 4 + 1,
            Self::I64(_) => 8 + 1,
            Self::I128(_) => 16 + 1,
            Self::F32(_) => 4 + 1,
            Self::F64(_) => 8 + 1,
            Self::Label(_) => 8 + 1,
            Self::Addr(adr) => 1 + adr.mc_size_in_bytes(),
            Self::Data(_) => 8 + 1,
            Self::Register(_) => 1,
            Self::Offset(_, _) => 1 + 8 + 1,
            Self::Unknown => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash)]
#[repr(u8)]
pub enum InstrSig {
    None = 0,
    Val,
    Adr,
    ValVal,
    ValAdr,
    AdrVal,
    AdrAdr,
}

impl InstrSig {
    pub const fn n_args(self) -> usize {
        match self {
            Self::None => 0,
            Self::Val | Self::Adr => 1,
            _ => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0,
    Und,
    Hlt,
    Mov,
    Push,
    Pop,
    Printi,
    Printc,
    Add,
    Sub,
    Mul,
    Div,
    Sdiv,
    Mod,
    Smod,
    And,
    Or,
    Xor,
    Cmp,
    Scmp,
    Fcmp,
    Jmp,
    Jgt,
    Jlt,
    Jeq,
    Jne,
    Juno,
    Junoeq,
    Junone,
    Junolt,
    Junogt,
    Junole,
    Junoge,
    Jord,
    Jordeq,
    Jordne,
    Jordlt,
    Jordgt,
    Jordle,
    Jordge,
    Call,
    Ret,
    Shl,
    Shr,
    Sshr,
    Sext,
}

impl Opcode {
    pub fn mc_repr(self) -> u8 {
        self as u8
    }

    pub fn n_args(self) -> usize {
        match self {
            Opcode::Nop => 0,
            Opcode::Und => 0,
            Opcode::Hlt => 0,
            Opcode::Mov => 2,
            Opcode::Push => 1,
            Opcode::Pop => 1,
            Opcode::Printi => 1,
            Opcode::Printc => 1,
            Opcode::Add => 2,
            Opcode::Sub => 2,
            Opcode::Mul => 2,
            Opcode::Div => 2,
            Opcode::Sdiv => 2,
            Opcode::Mod => 2,
            Opcode::Smod => 2,
            Opcode::And => 2,
            Opcode::Or => 2,
            Opcode::Xor => 2,
            Opcode::Cmp => 2,
            Opcode::Scmp => 2,
            Opcode::Fcmp => 2,
            Opcode::Jmp => 1,
            Opcode::Jgt => 1,
            Opcode::Jlt => 1,
            Opcode::Jeq => 1,
            Opcode::Jne => 1,
            Opcode::Juno => 1,
            Opcode::Junoeq => 1,
            Opcode::Junone => 1,
            Opcode::Junolt => 1,
            Opcode::Junogt => 1,
            Opcode::Junole => 1,
            Opcode::Junoge => 1,
            Opcode::Jord => 1,
            Opcode::Jordeq => 1,
            Opcode::Jordne => 1,
            Opcode::Jordlt => 1,
            Opcode::Jordgt => 1,
            Opcode::Jordle => 1,
            Opcode::Jordge => 1,
            Opcode::Call => 1,
            Opcode::Ret => 0,
            Opcode::Shl => 2,
            Opcode::Shr => 2,
            Opcode::Sshr => 2,
            Opcode::Sext => 1,
        }
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Nop => "nop",
            Self::Und => "und",
            Self::Hlt => "hlt",
            Self::Mov => "mov",
            Self::Push => "push",
            Self::Pop => "pop",
            Self::Printi => "printi",
            Self::Printc => "printc",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Sdiv => "sdiv",
            Self::Mod => "mod",
            Self::Smod => "smod",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Cmp => "cmp",
            Self::Scmp => "scmp",
            Self::Fcmp => "fcmp",
            Self::Jmp => "jmp",
            Self::Jgt => "jgt",
            Self::Jlt => "jlt",
            Self::Jeq => "jeq",
            Self::Jne => "jne",
            Self::Juno => "juno",
            Self::Junoeq => "junoeq",
            Self::Junone => "junone",
            Self::Junolt => "junolt",
            Self::Junogt => "junogt",
            Self::Junole => "junole",
            Self::Junoge => "junogt",
            Self::Jord => "jord",
            Self::Jordeq => "jordeq",
            Self::Jordne => "jordne",
            Self::Jordlt => "jordlt",
            Self::Jordgt => "jordgt",
            Self::Jordle => "jordle",
            Self::Jordge => "jordge",
            Self::Call => "call",
            Self::Ret => "ret",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Sshr => "sshr",
            Self::Sext => "sext",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub enum InstrSize {
    Unsized,
    I8,
    I16,
    I32,
    I64,
    I128,
    F32,
    F64,
}

impl Display for InstrSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsized => Ok(()),
            Self::I8 => write!(f, "i8"),
            Self::I16 => write!(f, "i16"),
            Self::I32 => write!(f, "i32"),
            Self::I64 => write!(f, "i64"),
            Self::I128 => write!(f, "i128"),
            Self::F32 => write!(f, "f32"),
            Self::F64 => write!(f, "f64"),
        }
    }
}

impl InstrSize {
    pub const fn mc_repr(self) -> u8 {
        match self {
            Self::Unsized => Unsized::MACHINE_CODE,
            Self::I8 => u8::MACHINE_CODE,
            Self::I16 => u16::MACHINE_CODE,
            Self::I32 => u32::MACHINE_CODE,
            Self::I64 => u64::MACHINE_CODE,
            Self::I128 => u128::MACHINE_CODE,
            Self::F32 => f32::MACHINE_CODE,
            Self::F64 => f64::MACHINE_CODE,
        }
    }

    pub const fn asm_repr(self) -> &'static str {
        match self {
            Self::Unsized => Unsized::ASM,
            Self::I8 => u8::ASM,
            Self::I16 => u16::ASM,
            Self::I32 => u32::ASM,
            Self::I64 => u64::ASM,
            Self::I128 => u128::ASM,
            Self::F32 => f32::ASM,
            Self::F64 => f64::ASM,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Instr {
    pub opcode: Opcode,
    pub size: InstrSize,
    pub arg0: Option<Token>,
    pub arg1: Option<Token>,
}

impl Display for Instr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.opcode, self.size)?;
        if let Some(ref arg) = self.arg0 {
            write!(f, " {}", arg)?;
        }
        if let Some(ref arg) = self.arg1 {
            write!(f, " {}", arg)?;
        }

        Ok(())
    }
}

impl Instr {
    pub fn signature(&self) -> InstrSig {
        let arg0 = if let Some(arg0) = &self.arg0 {
            match arg0 {
                Token::Addr(_) | Token::Label(_) | Token::Data(_) => InstrSig::Adr,
                _ => InstrSig::Val,
            }
        } else {
            InstrSig::None
        };
        let arg1 = if let Some(arg1) = &self.arg1 {
            match arg1 {
                Token::Addr(_) | Token::Label(_) | Token::Data(_) => InstrSig::Adr,
                _ => InstrSig::Val,
            }
        } else {
            InstrSig::None
        };
        match (arg0, arg1) {
            (InstrSig::Val, InstrSig::None) => InstrSig::Val,
            (InstrSig::Adr, InstrSig::None) => InstrSig::Adr,
            (InstrSig::Val, InstrSig::Val) => InstrSig::ValVal,
            (InstrSig::Val, InstrSig::Adr) => InstrSig::ValAdr,
            (InstrSig::Adr, InstrSig::Val) => InstrSig::AdrVal,
            (InstrSig::Adr, InstrSig::Adr) => InstrSig::AdrAdr,
            _ => unreachable!()
        }
    }

    pub const fn mc_size_in_bytes(&self) -> usize {
        let mut total = 2;
        if let Some(ref arg) = self.arg0 {
            total += arg.mc_size_in_bytes();
        }
        if let Some(ref arg) = self.arg1 {
            total += arg.mc_size_in_bytes();
        }
        total
    }
}


pub const ALL_REGS: &[Register] = &[
    Register::Rz,
    Register::Ra,
    Register::Rb,
    Register::Rc,
    Register::Rd,
    Register::Re,
    Register::Rf,
    Register::Rg,
    Register::Rh,
    Register::Ri,
    Register::Rj,
    Register::Rk,
    Register::Rl,
    Register::Bp,
    Register::Sp,
    Register::Pc,
    Register::Fl,
];

#[derive(Debug, Clone, Copy, Hash)]
#[repr(u8)]
pub enum Register {
    Rz = 0,
    Ra,
    Rb,
    Rc,
    Rd,
    Re,
    Rf,
    Rg,
    Rh,
    Ri,
    Rj,
    Rk,
    Rl,
    Bp,
    Sp,
    Pc,
    Fl,
}

impl Register {
    pub fn mc_repr(self) -> u8 {
        self as u8
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Rz => "rz",
            Self::Ra => "ra",
            Self::Rb => "rb",
            Self::Rc => "rc",
            Self::Rd => "rd",
            Self::Re => "re",
            Self::Rf => "rf",
            Self::Rg => "rg",
            Self::Rh => "rh",
            Self::Ri => "ri",
            Self::Rj => "rj",
            Self::Rk => "rk",
            Self::Rl => "rl",
            Self::Bp => "bp",
            Self::Sp => "sp",
            Self::Pc => "pc",
            Self::Fl => "fl",
        };
        write!(f, "{}", s)
    }
}
