use anyhow::Result;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, one_of, space0, space1},
    combinator::{opt, recognize, map, value},
    multi::{many0, many1},
    sequence::{preceded, terminated, tuple},
    IResult, error::Error,
};


use crate::k4s::{InstrSig, InstrSize, Register, Token, Opcode, Primitive, contexts::asm::AssemblyContext, Instr, Label, Linkage, Data};

use super::machine::tags::{LITERAL, REGISTER_OFFSET, ADDRESS};


fn decimal(size: InstrSize, input: &str) -> IResult<&str, Token> {
    map(many1(terminated(one_of("0123456789"), many0(char('_')))), |res| match size {
        InstrSize::I8 => Token::I8(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::I16 => Token::I16(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::I32 => Token::I32(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::I64 => Token::I64(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::I128 => Token::I128(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::F32 => Token::F32(res.into_iter().collect::<String>().parse().unwrap()),
        InstrSize::F64 => Token::F64(res.into_iter().collect::<String>().parse().unwrap()),
        _ => unimplemented!(),
    })(input)
}

fn hexadecimal(size: InstrSize, input: &str) -> IResult<&str, Token> {
    map(preceded(
        alt((tag::<&str, &str, Error<&str>>("0x"), tag("0X"))),
        recognize(many1(terminated(
            one_of("0123456789abcdefABCDEF"),
            many0(char('_')),
        ))),
    ), |res| match size {
        InstrSize::I8 => Token::I8(u8::from_str_radix(res, 16).unwrap()),
        InstrSize::I16 => Token::I16(u16::from_str_radix(res, 16).unwrap()),
        InstrSize::I32 => Token::I32(u32::from_str_radix(res, 16).unwrap()),
        InstrSize::I64 => Token::I64(u64::from_str_radix(res, 16).unwrap()),
        InstrSize::I128 => Token::I128(u128::from_str_radix(res, 16).unwrap()),
        _ => unimplemented!(),
    })(input)
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

pub fn literal(size: InstrSize, i: &str) -> IResult<&str, Token> {
    map(tuple((tag("$"), alt((|i| hexadecimal(size, i), |i| decimal(size, i))))), |res| res.1)(i)
}

pub fn header(i: &str) -> IResult<&str, &str> {
    recognize(tuple((
        tag("!"),
        many1(alt((alpha1, tag("_"), tag("."), recognize(|i| decimal(InstrSize::I64, i))))),
    )))(i)
}

pub fn label(i: &str) -> IResult<&str, Token> {
    map(tuple((
        tag("%"),
        many1(alt((alpha1, tag("_"), tag("."), recognize(|i| decimal(InstrSize::I64, i))))),
    )), |res| Token::Label(Label { name: res.1.join(""), linkage: Linkage::NeedsLinking }))(i)
}

pub fn data(i: &str) -> IResult<&str, Token> {
    map(tuple((
        tag("@"),
        many1(alt((alpha1, tag("_"), tag("."), recognize(|i| decimal(InstrSize::I64, i))))),
    )), |res| Token::Data(Data { label: Label { name: res.1.join(""), linkage: Linkage::NeedsLinking }, data: res.0.as_bytes().to_vec()} ))(i)
}

pub fn offset(i: &str) -> IResult<&str, Token> {
    map(tuple((opt(tag("-")), |i| decimal(InstrSize::I64, i), tag("+"), register)), |res| {
        if let Token::I64(off) = res.1 {
            if let Token::Register(reg) = res.3 {
                if res.0.is_some() {
                    Token::Offset(-(off as i64), reg)
                }
                else {
                    Token::Offset(off as i64, reg)
                }
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    })(i)
}

pub fn val(size: InstrSize, i: &str) -> IResult<&str, Token> {
    alt((register, |i| literal(size, i), label, data))(i)
}

pub fn addr(i: &str) -> IResult<&str, Token> {
    map(tuple((tag("["), alt((|a| val(InstrSize::I64, a), offset)), tag("]"))), |res| {
        Token::Addr(Box::new(res.1))
    })(i)
}


pub fn token(size: InstrSize, asm: &str) -> IResult<&str, Token>{
    alt((|a| val(size, a), addr))(asm)
}

pub fn opcode(asm: &str) -> IResult<&str, Opcode> {
    alt((
        alt((
            |asm| Opcode::Und.parse_asm(asm),
            |asm| Opcode::Hlt.parse_asm(asm),
            |asm| Opcode::Mov.parse_asm(asm),
            |asm| Opcode::Push.parse_asm(asm),
            |asm| Opcode::Pop.parse_asm(asm),
            |asm| Opcode::Printi.parse_asm(asm),
            |asm| Opcode::Printc.parse_asm(asm),
            |asm| Opcode::Add.parse_asm(asm),
            |asm| Opcode::Sub.parse_asm(asm),
            |asm| Opcode::Mul.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Div.parse_asm(asm),
            |asm| Opcode::Sdiv.parse_asm(asm),
            |asm| Opcode::Mod.parse_asm(asm),
            |asm| Opcode::Smod.parse_asm(asm),
            |asm| Opcode::And.parse_asm(asm),
            |asm| Opcode::Or.parse_asm(asm),
            |asm| Opcode::Xor.parse_asm(asm),
            |asm| Opcode::Cmp.parse_asm(asm),
            |asm| Opcode::Scmp.parse_asm(asm),
            |asm| Opcode::Fcmp.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Jmp.parse_asm(asm),
            |asm| Opcode::Jgt.parse_asm(asm),
            |asm| Opcode::Jlt.parse_asm(asm),
            |asm| Opcode::Jeq.parse_asm(asm),
            |asm| Opcode::Jne.parse_asm(asm),
            |asm| Opcode::Juno.parse_asm(asm),
            |asm| Opcode::Junoeq.parse_asm(asm),
            |asm| Opcode::Junone.parse_asm(asm),
            |asm| Opcode::Junolt.parse_asm(asm),
            |asm| Opcode::Junogt.parse_asm(asm),
            |asm| Opcode::Junole.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Junoge.parse_asm(asm),
            |asm| Opcode::Jord.parse_asm(asm),
            |asm| Opcode::Jordeq.parse_asm(asm),
            |asm| Opcode::Jordne.parse_asm(asm),
            |asm| Opcode::Jordlt.parse_asm(asm),
            |asm| Opcode::Jordgt.parse_asm(asm),
            |asm| Opcode::Jordle.parse_asm(asm),
            |asm| Opcode::Jordge.parse_asm(asm),
            |asm| Opcode::Call.parse_asm(asm),
            |asm| Opcode::Ret.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Shl.parse_asm(asm),
            |asm| Opcode::Shr.parse_asm(asm),
            |asm| Opcode::Sshr.parse_asm(asm),
            |asm| Opcode::Sext.parse_asm(asm),
        )),
    ))(asm)
}

pub fn size(asm: &str) -> IResult<&str, InstrSize> {
    alt((
        |asm| value(InstrSize::I8, tag("i8"))(asm),
        |asm| value(InstrSize::I16, tag("i16"))(asm),
        |asm| value(InstrSize::I32, tag("i32"))(asm),
        |asm| value(InstrSize::I64, tag("i64"))(asm),
        |asm| value(InstrSize::I128, tag("i128"))(asm),
        |asm| value(InstrSize::F32, tag("f32"))(asm),
        |asm| value(InstrSize::F64, tag("f64"))(asm),
    ))(asm)
}

impl Token {
    pub fn assemble(&mut self, ctx: &mut AssemblyContext) {
        match self {
            Token::I8(val) => ctx.push_program_bytes(&[LITERAL, InstrSize::I8.mc_repr(), *val]),
            Token::I16(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::I32(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::I64(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::I128(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::F32(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::F64(val) => {
                ctx.push_program_bytes(&[LITERAL]);
                ctx.push_program_bytes(&val.to_bytes());
            },
            Token::Register(reg) => {
                ctx.push_program_bytes(&[reg.mc_repr()]);
            }
            Token::Offset(off, reg) => {
                ctx.push_program_bytes(&[REGISTER_OFFSET]);
                ctx.push_program_bytes(&(*off as u64).to_bytes());
                ctx.push_program_bytes(&[reg.mc_repr()]);
            }
            Token::Data(dat) => {
                if let Some(linked_location) = ctx.linked_refs.get(&dat.label) {
                    let linked_location = *linked_location;
                    dat.label.linkage = Linkage::Linked(linked_location);
                    ctx.push_program_bytes(&[LITERAL]);
                    ctx.push_program_bytes(&linked_location.to_bytes());
                } else {
                    ctx.push_program_bytes(&[LITERAL]);
                    ctx.unlinked_refs.insert(dat.label.to_owned(), ctx.pc);
                    ctx.push_program_bytes(&[0; 8])
                }
            },
            Token::Label(lab) => {
                if let Some(linked_location) = ctx.linked_refs.get(lab) {
                    let linked_location = *linked_location;
                    lab.linkage = Linkage::Linked(linked_location);
                    ctx.push_program_bytes(&[LITERAL]);
                    ctx.push_program_bytes(&linked_location.to_bytes());
                } else {
                    ctx.push_program_bytes(&[LITERAL]);
                    ctx.unlinked_refs.insert(lab.to_owned(), ctx.pc);
                    ctx.push_program_bytes(&[0; 8])
                }
            }
            Token::Addr(adr) => {
                ctx.push_program_bytes(&[ADDRESS]);
                adr.assemble(ctx);
            }
            Token::Unknown => panic!("Attempt to assemble an unknown token") // todo: Err instead of panic
        }
    }
}

impl InstrSig {
    pub fn match_asm(self, size: InstrSize, asm: &str) -> IResult<&str, &str> {
        match self {
            Self::Val => recognize(|i| val(size, i))(asm),
            Self::Adr => recognize(addr)(asm),
            Self::ValVal => recognize(tuple((|i| val(size, i), space1, |i| val(size, i))))(asm),
            Self::ValAdr => recognize(tuple((|i| val(size, i), space1, addr)))(asm),
            Self::AdrVal => recognize(tuple((addr, space1, |i| val(size, i))))(asm),
            Self::AdrAdr => recognize(tuple((addr, space1, addr)))(asm),
            Self::None => Ok(("", asm)),
        }
    }
}

impl Opcode {
    pub fn parse_asm(self, asm: &str) -> IResult<&str, Opcode> {
        value(self, tag(format!("{}", self).as_bytes()))(asm)
    }
}

impl Instr {
    pub const fn new(opcode: Opcode, size: InstrSize, arg0: Option<Token>, arg1: Option<Token>) -> Self {
        Self { opcode, size, arg0, arg1 }
    }
    
    pub fn match_asm(self, asm: &str) -> IResult<&str, &str> {
        let sig = |i| self.signature().match_asm(self.size, i);
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

    pub fn assemble(&mut self, ctx: &mut AssemblyContext) -> Result<()> {
        let opcode = self.opcode.mc_repr();
        let size = self.size.mc_repr();
        ctx.push_program_bytes(&[opcode, size]);
        if let Some(ref mut arg) = self.arg0 {
            arg.assemble(ctx);
        }
        if let Some(ref mut arg) = self.arg1{
            arg.assemble(ctx);
        }
        
        Ok(())
    }
}