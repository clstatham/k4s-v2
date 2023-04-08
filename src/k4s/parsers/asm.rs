use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, one_of, space0, space1},
    combinator::{opt, recognize, map, value},
    multi::{many0, many1},
    sequence::{preceded, terminated, tuple},
    IResult, error::Error,
};


use crate::k4s::{InstrSig, InstrSize, Register, Token, Opcode, Primitive, contexts::asm::AssemblyContext};

use super::machine::tags::{LITERAL, REGISTER_OFFSET};


fn decimal(input: &str) -> IResult<&str, Token> {
    map(many1(terminated(one_of("0123456789"), many0(char('_')))), |res| Token::I128(res.into_iter().collect::<String>().parse().unwrap()))(input)
}

fn hexadecimal(input: &str) -> IResult<&str, Token> {
    map(preceded(
        alt((tag::<&str, &str, Error<&str>>("0x"), tag("0X"))),
        recognize(many1(terminated(
            one_of("0123456789abcdefABCDEF"),
            many0(char('_')),
        ))),
    ), |res| Token::I128(u128::from_str_radix(res, 16).unwrap()))(input)
}

fn register(i: &str) -> IResult<&str, Token> {
    map(alt((
        value(Register::Rz, tag("rz")),
        value(Register::Ra, tag("ra")),
        value(Register::Rb, tag("rb")),
        value(Register::Rc, tag("rc")),
        value(Register::Rd, tag("rd")),
        value(Register::Re, tag("re")),
        value(Register::Rf, tag("rf")),
        value(Register::Rg, tag("rg")),
        value(Register::Rh, tag("rh")),
        value(Register::Ri, tag("ri")),
        value(Register::Rj, tag("rj")),
        value(Register::Rk, tag("rk")),
        value(Register::Rl, tag("rl")),
        value(Register::Bp, tag("bp")),
        value(Register::Sp, tag("sp")),
        value(Register::Pc, tag("pc")),
        value(Register::Fl, tag("fl")),
    )), Token::Register)(i)
}

pub fn literal(i: &str) -> IResult<&str, Token> {
    map(tuple((tag("$"), alt((hexadecimal, decimal)))), |res| res.1)(i)
}

pub fn label(i: &str) -> IResult<&str, Token> {
    map(tuple((
        tag("%"),
        many1(alt((alpha1, tag("_"), tag("."), recognize(decimal)))),
    )), |res| Token::Label(res.1.join("")))(i)
}

pub fn data(i: &str) -> IResult<&str, Token> {
    map(tuple((
        tag("@"),
        many1(alt((alpha1, tag("_"), tag("."), recognize(decimal)))),
    )), |res| Token::Data(res.1.join("")))(i)
}

pub fn offset(i: &str) -> IResult<&str, Token> {
    map(preceded(opt(tag("-")), tuple((decimal, tag("+"), register))), |res| {
        if let Token::I128(off) = res.0 {
            if let Token::Register(reg) = res.2 {
                Token::Offset(off as u64, reg)
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    })(i)
}

pub fn val(i: &str) -> IResult<&str, Token> {
    alt((register, literal, label, data))(i)
}

pub fn addr(i: &str) -> IResult<&str, Token> {
    map(tuple((tag("["), alt((val, offset)), tag("]"))), |res| {
        Token::Addr(Box::new(res.1))
    })(i)
}

impl Token {
    pub fn assemble(self, ctx: &mut AssemblyContext) -> Vec<u8> {
        match self {
            Token::I8(val) => vec![LITERAL, InstrSize::I8.mc_repr(), val],
            Token::I16(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::I32(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::I64(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::I128(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::F32(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::F64(val) => {
                let mut out = vec![LITERAL];
                out.extend_from_slice(&val.to_bytes());
                out
            },
            Token::Register(reg) => {
                vec![reg.mc_repr()]
            }
            Token::Offset(off, reg) => {
                let mut out = vec![REGISTER_OFFSET];
                out.extend_from_slice(&off.to_bytes());
                out.push(reg.mc_repr());
                out
            }
            _ => todo!()
        }
    }
}

impl InstrSig {
    pub fn match_asm(self, asm: &str) -> IResult<&str, &str> {
        match self {
            Self::Val => recognize(val)(asm),
            Self::Adr => recognize(addr)(asm),
            Self::ValVal => recognize(tuple((val, space1, val)))(asm),
            Self::ValAdr => recognize(tuple((val, space1, addr)))(asm),
            Self::AdrVal => recognize(tuple((addr, space1, val)))(asm),
            Self::AdrAdr => recognize(tuple((addr, space1, addr)))(asm),
            Self::None => Ok(("", asm)),
        }
    }

    pub fn assemble<'asm>(self, asm: &'asm str, ctx: &mut AssemblyContext) -> IResult<&'asm str, Vec<u8>> {
        match self {
            Self::Val => map(val, |res| res.assemble(ctx))(asm),
            Self::Adr => map(addr, |res| res.assemble(ctx))(asm),
            Self::ValVal => map(tuple((val, space1, val)), |(val1, _, val2)| {
                let mut out = val1.assemble(ctx);
                out.extend_from_slice(&val2.assemble(ctx));
                out
            })(asm),
            Self::ValAdr => map(tuple((val, space1, addr)), |(val, _, adr)| {
                let mut out = val.assemble(ctx);
                out.extend_from_slice(&adr.assemble(ctx));
                out
            })(asm),
            Self::AdrVal => map(tuple((addr, space1, val)), |(adr, _, val)| {
                let mut out = adr.assemble(ctx);
                out.extend_from_slice(&val.assemble(ctx));
                out
            })(asm),
            Self::AdrAdr => map(tuple((addr, space1, addr)), |(adr1, _, adr2)| {
                let mut out = adr1.assemble(ctx);
                out.extend_from_slice(&adr2.assemble(ctx));
                out
            })(asm),
            Self::None => Ok(("", vec![])),
        }
    }
}

impl Opcode {
    pub fn parse_asm(self, asm: &str) -> IResult<&str, Opcode> {
        value(self, tag(format!("{}", self).as_bytes()))(asm)
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct AsmInstr {
    pub opcode: Opcode,
    pub size: InstrSize,
    pub signature: InstrSig,
}

impl AsmInstr {
    pub const fn new(opcode: Opcode, size: InstrSize, signature: InstrSig) -> Self {
        Self { opcode, size, signature }
    }

    pub const fn as_machine_code(self) -> [u8; 3] {
        [self.opcode as u8, self.signature as u8, self.size.mc_repr()]
    }
}

impl AsmInstr {
    pub fn match_asm(self, asm: &str) -> IResult<&str, &str> {
        let sig = |i| self.signature.match_asm(i);
        let opcode = format!("{}", self.opcode);
        let opcode = opcode.as_str();
        if let InstrSize::Unsized = self.size {
            recognize(tuple((tag(opcode), space0, sig)))(asm)
        } else {
            let size = self.size.asm_repr();
            recognize(tuple((
                tag(opcode),
                space1,
                tag(size),
                space1,
                sig,
            )))(asm)
        }
    }

    pub fn assemble<'asm>(self, asm: &'asm str, ctx: &mut AssemblyContext) -> IResult<&'asm str, Vec<u8>> {
        let sig = |i| self.signature.assemble(i, ctx);
        let opcode = |i| self.opcode.parse_asm(i);
        if let InstrSize::Unsized = self.size {
            let (rest, (opcode, _, sig)) = tuple((opcode, space0, sig))(asm)?;
            let mut out = vec![opcode.mc_repr()];
            out.extend_from_slice(&sig);
            Ok((rest, out))
        } else {
            let size = self.size.asm_repr();
            let (rest, (opcode, _, _, _, sig)) = tuple((
                opcode,
                space1,
                tag(size),
                space1,
                sig,
            ))(asm)?;
            let mut out = vec![opcode.mc_repr(), self.size.mc_repr()];
            out.extend_from_slice(&sig);
            Ok((rest, out))
        }
    }
}