use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt::Display,
    ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub},
};

use anyhow::{Error, Result};

pub mod contexts;
pub mod parsers;

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

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Default)]
pub struct Label {
    name: String,
    pub region_id: Option<usize>,
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.name())
    }
}

impl Label {
    pub fn new(label: String) -> Self {
        Self {
            name: label,
            region_id: None,
        }
    }

    pub fn new_in_region(label: String, region_id: usize) -> Self {
        Self {
            name: label,
            region_id: Some(region_id),
        }
    }

    pub fn name(&self) -> String {
        rustc_demangle::demangle(&self.name)
            .as_str()
            .replace(['$', '.', '(', ')', '[', ']', '*', '{', '}', ','], "_")
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Hash)]
pub struct Data {
    pub label: Label,
    pub align: usize,
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
    RegOffset(i64, Register),
    Register(Register),
    Label(Label),
    Data(Data),
    LabelOffset(i64, Label),
}

impl PartialOrd for Token {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some(a.cmp(b)),
            (Self::I16(a), Self::I16(b)) => Some(a.cmp(b)),
            (Self::I32(a), Self::I32(b)) => Some(a.cmp(b)),
            (Self::I64(a), Self::I64(b)) => Some(a.cmp(b)),
            (Self::I128(a), Self::I128(b)) => Some(a.cmp(b)),
            (Self::F32(a), Self::F32(b)) => a.partial_cmp(b),
            (Self::F64(a), Self::F64(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => a.eq(b),
            (Self::I16(a), Self::I16(b)) => a.eq(b),
            (Self::I32(a), Self::I32(b)) => a.eq(b),
            (Self::I64(a), Self::I64(b)) => a.eq(b),
            (Self::I128(a), Self::I128(b)) => a.eq(b),
            (Self::F32(a), Self::F32(b)) => a.eq(b),
            (Self::F64(a), Self::F64(b)) => a.eq(b),
            _ => false,
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown => write!(f, "(UNKNOWN)"),
            Self::I8(v) => write!(f, "${}", v),
            Self::I16(v) => write!(f, "${}", v),
            Self::I32(v) => write!(f, "${}", v),
            Self::I64(v) => write!(f, "${}", v),
            Self::I128(v) => write!(f, "${}", v),
            Self::F32(v) => write!(f, "${}", v),
            Self::F64(v) => write!(f, "${}", v),
            Self::Addr(v) => write!(f, "[{}]", v),
            Self::RegOffset(off, reg) => write!(f, "[{}+{}]", *off, reg),
            Self::Register(reg) => write!(f, "{}", reg),
            Self::Label(lab) => write!(f, "{}", lab),
            Self::Data(dat) => write!(f, "@{}", dat.label.name()),
            Self::LabelOffset(off, lab) => {
                write!(f, "[{}+@{}]", *off, lab.name())
            }
        }
    }
}

macro_rules! token_arith_impl_checked {
    ($ins:ident, $op:ident) => {
        pub fn $ins(&self, rhs: &Self) -> Result<Self> {
            match (self, rhs) {
                (Self::I8(a), Self::I8(b)) => {
                    Ok(Self::I8(a.$op(*b).ok_or(Error::msg("over/underflow"))?))
                }
                (Self::I16(a), Self::I16(b)) => {
                    Ok(Self::I16(a.$op(*b).ok_or(Error::msg("over/underflow"))?))
                }
                (Self::I32(a), Self::I32(b)) => {
                    Ok(Self::I32(a.$op(*b).ok_or(Error::msg("over/underflow"))?))
                }
                (Self::I64(a), Self::I64(b)) => {
                    Ok(Self::I64(a.$op(*b).ok_or(Error::msg("over/underflow"))?))
                }
                (Self::I128(a), Self::I128(b)) => {
                    Ok(Self::I128(a.$op(*b).ok_or(Error::msg("over/underflow"))?))
                }
                // (Self::F32(a), Self::F32(b)) => Ok(Self::F32(a.$op(b))),
                // (Self::F64(a), Self::F64(b)) => Ok(Self::F64(a.$op(b))),
                _ => Err(Error::msg(
                    "token types must match and be numeric for arithmetic operations",
                )),
            }
        }
    };
}

macro_rules! token_int_arith_impl {
    ($ins:ident, $op:ident) => {
        pub fn $ins(&self, rhs: &Self) -> Result<Self> {
            match (self, rhs) {
                (Self::I8(a), Self::I8(b)) => Ok(Self::I8(a.$op(b))),
                (Self::I16(a), Self::I16(b)) => Ok(Self::I16(a.$op(b))),
                (Self::I32(a), Self::I32(b)) => Ok(Self::I32(a.$op(b))),
                (Self::I64(a), Self::I64(b)) => Ok(Self::I64(a.$op(b))),
                (Self::I128(a), Self::I128(b)) => Ok(Self::I128(a.$op(b))),
                _ => Err(Error::msg(
                    "token types must match and be integers for integer arithmetic operations",
                )),
            }
        }
    };
}

macro_rules! token_signed_int_arith_impl {
    ($ins:ident, $op:ident) => {
        pub fn $ins(&self, rhs: &Self) -> Result<Self> {
            match (self, rhs) {
                (Self::I8(a), Self::I8(b)) => Ok(Self::I8((*a as i8).$op(*b as i8) as u8)),
                (Self::I16(a), Self::I16(b)) => Ok(Self::I16((*a as i16).$op(*b as i16) as u16)),
                (Self::I32(a), Self::I32(b)) => Ok(Self::I32((*a as i32).$op(*b as i32) as u32)),
                (Self::I64(a), Self::I64(b)) => Ok(Self::I64((*a as i64).$op(*b as i64) as u64)),
                (Self::I128(a), Self::I128(b)) => {
                    Ok(Self::I128((*a as i128).$op(*b as i128) as u128))
                }
                _ => Err(Error::msg(
                    "token types must match and be integers for signed integer arithmetic operations",
                )),
            }
        }
    };
}

impl Token {
    token_arith_impl_checked!(add, checked_add);
    token_arith_impl_checked!(sub, checked_sub);
    token_arith_impl_checked!(mul, checked_mul);
    token_arith_impl_checked!(div, checked_div);
    token_arith_impl_checked!(rem, checked_rem);
    token_int_arith_impl!(bitand, bitand);
    token_int_arith_impl!(bitor, bitor);
    token_int_arith_impl!(bitxor, bitxor);
    token_int_arith_impl!(shl, shl);
    token_int_arith_impl!(shr, shr);
    token_signed_int_arith_impl!(sshr, shr);
    token_signed_int_arith_impl!(smod, rem);
    token_signed_int_arith_impl!(sdiv, div);

    pub fn sext(&self, size: InstrSize) -> Option<Self> {
        match (self, size) {
            (Self::I8(_v), InstrSize::I16) => self
                .as_signed_integer::<i8>()
                .map(|v| Self::I16(v as i16 as u16)),
            (Self::I8(_v), InstrSize::I32) => self
                .as_signed_integer::<i8>()
                .map(|v| Self::I32(v as i32 as u32)),
            (Self::I8(_v), InstrSize::I64) => self
                .as_signed_integer::<i8>()
                .map(|v| Self::I64(v as i64 as u64)),
            (Self::I8(_v), InstrSize::I128) => self
                .as_signed_integer::<i8>()
                .map(|v| Self::I128(v as i128 as u128)),

            (Self::I16(_v), InstrSize::I32) => self
                .as_signed_integer::<i16>()
                .map(|v| Self::I32(v as i32 as u32)),
            (Self::I16(_v), InstrSize::I64) => self
                .as_signed_integer::<i16>()
                .map(|v| Self::I64(v as i64 as u64)),
            (Self::I16(_v), InstrSize::I128) => self
                .as_signed_integer::<i16>()
                .map(|v| Self::I128(v as i128 as u128)),

            (Self::I32(_v), InstrSize::I64) => self
                .as_signed_integer::<i32>()
                .map(|v| Self::I64(v as i64 as u64)),
            (Self::I32(_v), InstrSize::I128) => self
                .as_signed_integer::<i32>()
                .map(|v| Self::I128(v as i128 as u128)),

            (Self::I64(_v), InstrSize::I128) => self
                .as_signed_integer::<i64>()
                .map(|v| Self::I128(v as i128 as u128)),

            _ => None,
        }
    }

    pub fn as_integer<T>(&self) -> Option<T>
    where
        T: TryFrom<u128> + TryFrom<u64> + TryFrom<u32> + TryFrom<u16> + TryFrom<u8>,
    {
        match self {
            Self::I128(v) => (*v).try_into().ok(),
            Self::I64(v) => (*v).try_into().ok(),
            Self::I32(v) => (*v).try_into().ok(),
            Self::I16(v) => (*v).try_into().ok(),
            Self::I8(v) => (*v).try_into().ok(),
            _ => None,
        }
    }

    pub fn as_signed_integer<T>(&self) -> Option<T>
    where
        T: TryFrom<i128> + TryFrom<i64> + TryFrom<i32> + TryFrom<i16> + TryFrom<i8>,
    {
        match self {
            Self::I128(v) => (*v as i128).try_into().ok(),
            Self::I64(v) => (*v as i64).try_into().ok(),
            Self::I32(v) => (*v as i32).try_into().ok(),
            Self::I16(v) => (*v as i16).try_into().ok(),
            Self::I8(v) => (*v as i8).try_into().ok(),
            _ => None,
        }
    }

    pub fn scmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::I8(a), Self::I8(b)) => Some((*a as i8).cmp(&(*b as i8))),
            (Self::I16(a), Self::I16(b)) => Some((*a as i16).cmp(&(*b as i16))),
            (Self::I32(a), Self::I32(b)) => Some((*a as i32).cmp(&(*b as i32))),
            (Self::I64(a), Self::I64(b)) => Some((*a as i64).cmp(&(*b as i64))),
            (Self::I128(a), Self::I128(b)) => Some((*a as i128).cmp(&(*b as i128))),
            (Self::F32(a), Self::F32(b)) => a.partial_cmp(b),
            (Self::F64(a), Self::F64(b)) => a.partial_cmp(b),
            _ => None,
        }
    }

    pub fn from_integer_size<T>(t: T, size: InstrSize) -> Option<Self>
    where
        T: TryInto<u128> + TryInto<u64> + TryInto<u32> + TryInto<u16> + TryInto<u8>,
    {
        match size {
            InstrSize::I8 => t.try_into().ok().map(Self::I8),
            InstrSize::I16 => t.try_into().ok().map(Self::I16),
            InstrSize::I32 => t.try_into().ok().map(Self::I32),
            InstrSize::I64 => t.try_into().ok().map(Self::I64),
            InstrSize::I128 => t.try_into().ok().map(Self::I128),
            _ => None,
        }
    }

    pub fn from_fp_size<T>(t: T, size: InstrSize) -> Option<Self>
    where
        T: TryInto<f32> + TryInto<f64>,
    {
        match size {
            InstrSize::F32 => t.try_into().ok().map(Self::F32),
            InstrSize::F64 => t.try_into().ok().map(Self::F64),
            _ => None,
        }
    }

    pub fn instr_size(&self) -> InstrSize {
        InstrSize::from_integer_bits(self.value_size_in_bytes() as u32 * 8).unwrap()
    }

    pub fn display_with_symbols(&self, symbols: &BTreeMap<u64, String>) -> String {
        match self {
            Self::Addr(v) => format!("[{}]", v.display_with_symbols(symbols)),
            Self::I64(v) => {
                if let Some(lab) = symbols.get(v) {
                    lab.to_string()
                } else {
                    format!("${}", v)
                }
            }
            _ => format!("{}", self),
        }
    }

    pub const fn value_size_in_bytes(&self) -> usize {
        match self {
            Self::I8(_) => 1,
            Self::I16(_) => 2,
            Self::I32(_) => 4,
            Self::I64(_) => 8,
            Self::I128(_) => 16,
            Self::F32(_) => 4,
            Self::F64(_) => 8,
            Self::Label(_) => 8,
            Self::Addr(_) => 8,
            Self::Data(_) => 8,
            Self::Register(_) => 8,
            Self::RegOffset(_, _) => 8,
            Self::LabelOffset(_, _) => 8,
            Self::Unknown => 0,
        }
    }

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
            Self::RegOffset(_, _) => 1 + 8 + 1,
            Self::LabelOffset(_, _) => 8 + 1,
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
    Jmp,
    Jgt,
    Jlt,
    Jge,
    Jle,
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
    Enpt,
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
            Opcode::Jmp => 1,
            Opcode::Jgt => 1,
            Opcode::Jlt => 1,
            Opcode::Jeq => 1,
            Opcode::Jne => 1,
            Opcode::Jge => 1,
            Opcode::Jle => 1,
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
            Opcode::Enpt => 1,
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
            Self::Jmp => "jmp",
            Self::Jgt => "jgt",
            Self::Jlt => "jlt",
            Self::Jeq => "jeq",
            Self::Jne => "jne",
            Self::Jge => "jge",
            Self::Jle => "jle",
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
            Self::Enpt => "enpt",
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
    pub const fn in_bytes(self) -> usize {
        match self {
            Self::Unsized => 0,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::I128 => 16,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub const fn from_integer_bits(bits: u32) -> Option<Self> {
        match bits {
            1 => Some(Self::I8),
            8 => Some(Self::I8),
            16 => Some(Self::I16),
            32 => Some(Self::I32),
            64 => Some(Self::I64),
            128 => Some(Self::I128),
            _ => None,
        }
    }

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
    pub fn arg0(&self) -> Result<Token> {
        self.arg0
            .clone()
            .ok_or(Error::msg("no arg0 for instruction"))
    }

    pub fn arg1(&self) -> Result<Token> {
        self.arg1
            .clone()
            .ok_or(Error::msg("no arg1 for instruction"))
    }

    pub fn display_with_symbols(&self, symbols: &BTreeMap<u64, String>) -> String {
        use std::fmt::Write;
        let mut f = String::new();
        write!(f, "{} {}", self.opcode, self.size).unwrap();
        if let Some(ref arg) = self.arg0 {
            write!(f, " {}", arg.display_with_symbols(symbols)).unwrap();
        }
        if let Some(ref arg) = self.arg1 {
            write!(f, " {}", arg.display_with_symbols(symbols)).unwrap();
        }

        f
    }

    pub fn signature(&self) -> InstrSig {
        let arg0 = if let Some(arg0) = &self.arg0 {
            match arg0 {
                Token::Addr(_) | Token::Label(_) | Token::Data(_) | Token::LabelOffset(_, _) => {
                    InstrSig::Adr
                }
                _ => InstrSig::Val,
            }
        } else {
            InstrSig::None
        };
        let arg1 = if let Some(arg1) = &self.arg1 {
            match arg1 {
                Token::Addr(_) | Token::Label(_) | Token::Data(_) | Token::LabelOffset(_, _) => {
                    InstrSig::Adr
                }
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
            _ => unreachable!(),
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

pub const GP_REGS: &[Register] = &[
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
];

#[derive(Debug, Clone, Copy, Hash, PartialEq, PartialOrd, Eq, Ord)]
#[repr(u8)]
pub enum Register {
    R0 = 1,
    R1,
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
    Pt,
}

impl Register {
    pub fn mc_repr(self) -> u8 {
        self as u8
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::R0 => "r0",
            Self::R1 => "r1",
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
            Self::Pt => "pt",
        };
        write!(f, "{}", s)
    }
}
