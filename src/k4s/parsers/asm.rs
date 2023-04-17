use anyhow::Result;
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::{
        complete::{alpha1, char, hex_digit1, one_of, space1},
        streaming::alphanumeric1,
    },
    combinator::{map, opt, recognize, value},
    multi::{many0, many1},
    sequence::{preceded, terminated, tuple},
    IResult,
};

use crate::k4s::{
    contexts::asm::Header, Data, Instr, InstrSig, InstrSize, Label, Opcode, Primitive, Register,
    Token,
};

use super::machine::tags::{ADDRESS, LITERAL, REGISTER_OFFSET};

pub fn decimal(size: InstrSize, input: &str) -> IResult<&str, Token> {
    map(
        many1(terminated(
            one_of("0123456789"),
            many0(alt((char('_'), char('.')))),
        )),
        |res| match size {
            InstrSize::I8 => Token::I8(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing I8: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::I16 => Token::I16(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing I16: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::I32 => Token::I32(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing I32: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::I64 => Token::I64(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing I64: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::I128 => Token::I128(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing I128: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::F32 => Token::F32(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing F32: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            InstrSize::F64 => Token::F64(res.iter().collect::<String>().parse().unwrap_or_else(
                |err| {
                    panic!(
                        "Error parsing F64: {}: {err}",
                        res.into_iter().collect::<String>()
                    )
                },
            )),
            _ => unimplemented!(),
        },
    )(input)
}

pub fn hexadecimal(size: InstrSize, input: &str) -> IResult<&str, Token> {
    map(
        preceded(
            alt((tag("0x"), tag("0X"))),
            recognize(many1(terminated(
                one_of("0123456789abcdefABCDEF"),
                many0(char('_')),
            ))),
        ),
        |res| match size {
            InstrSize::I8 => Token::I8(u8::from_str_radix(res, 16).unwrap()),
            InstrSize::I16 => Token::I16(u16::from_str_radix(res, 16).unwrap()),
            InstrSize::I32 => Token::I32(u32::from_str_radix(res, 16).unwrap()),
            InstrSize::I64 => Token::I64(u64::from_str_radix(res, 16).unwrap()),
            InstrSize::I128 => Token::I128(u128::from_str_radix(res, 16).unwrap()),
            _ => unimplemented!(),
        },
    )(input)
}

fn register(i: &str) -> IResult<&str, Token> {
    map(
        alt((
            value(Register::R0, tag("r0")),
            value(Register::R1, tag("r1")),
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
            value(Register::Pt, tag("pt")),
        )),
        Token::Register,
    )(i)
}

pub fn literal(size: InstrSize, i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("$"),
            alt((
                map(tag("NaN"), |_nan| match size {
                    InstrSize::F32 => Token::F32(f32::NAN),
                    InstrSize::F64 => Token::F64(f64::NAN),
                    _ => unreachable!(),
                }),
                |i| hexadecimal(size, i),
                |i| decimal(size, i),
            )),
        )),
        |res| res.1,
    )(i)
}

pub fn header(i: &str) -> IResult<&str, &str> {
    recognize(tuple((
        tag("!"),
        many1(alt((
            alpha1,
            tag("_"),
            tag("."),
            recognize(|i| decimal(InstrSize::I64, i)),
        ))),
    )))(i)
}

pub fn header_entry(i: &str) -> IResult<&str, Header> {
    map(
        tuple((tag("!ent"), space1, label, space1, tag("@"), space1, |i| {
            hexadecimal(InstrSize::I64, i)
        })),
        |(_, _, lab, _, _, _, adr)| {
            if let Token::Label(lab) = lab {
                if let Token::I64(adr) = adr {
                    Header::Entry(lab, adr)
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
    )(i)
}

pub fn header_region(i: &str) -> IResult<&str, Header> {
    map(
        tuple((
            tag("!region"),
            space1,
            |i| hexadecimal(InstrSize::I64, i),
            space1,
            tag(">"),
            space1,
            |i| hexadecimal(InstrSize::I64, i),
            space1,
            tag(":"),
            many1(tuple((space1, alt((label, data_tag))))),
        )),
        |(_, _, virt, _, _, _, load, _, _, labels)| {
            if let Token::I64(virt) = virt {
                if let Token::I64(load) = load {
                    Header::Region(
                        virt,
                        load,
                        labels
                            .iter()
                            .map(|(_, label)| {
                                if let Token::Label(lab) = label {
                                    lab.to_owned()
                                } else if let Token::Data(dat) = label {
                                    dat.label.to_owned()
                                } else {
                                    unreachable!()
                                }
                            })
                            .collect(),
                    )
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
    )(i)
}

pub fn label(i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("%"),
            many1(alt((
                alpha1,
                tag("_"),
                tag("."),
                tag("$"),
                tag("-"),
                recognize(|i| decimal(InstrSize::I64, i)),
            ))),
        )),
        |(_, name)| Token::Label(Label::new(name.join(""))),
    )(i)
}

pub fn data_tag(i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("@"),
            many1(alt((
                alpha1,
                tag("_"),
                tag("."),
                tag("$"),
                tag("-"),
                recognize(|i| decimal(InstrSize::I64, i)),
            ))),
        )),
        |(_, label)| {
            Token::Data(Data {
                label: Label::new(label.join("")),
                align: 1,
                data: Vec::new(),
            })
        },
    )(i)
}

pub fn lab_offset_const(i: &str) -> IResult<&str, (Label, Token)> {
    map(
        tuple((
            data_tag,
            space1,
            tag("align8 ("),
            opt(tag("-")),
            |i| decimal(InstrSize::I64, i),
            tag("+"),
            data_tag,
            tag(")"),
        )),
        |(name, _, _, neg, off, _, lab, _)| {
            if let Token::I64(off) = off {
                if let Token::Data(data) = lab {
                    if let Token::Data(name) = name {
                        if neg.is_some() {
                            (name.label, Token::LabelOffset(-(off as i64), data.label))
                        } else {
                            (name.label, Token::LabelOffset(off as i64, data.label))
                        }
                    } else {
                        unreachable!()
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
    )(i)
}

pub fn data(i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("@"),
            many1(alt((alphanumeric1, tag("_"), tag("."), tag("$"), tag("-")))),
            space1,
            alt((
                value(1024, tag("align1024")),
                value(2048, tag("align2048")),
                value(4096, tag("align4096")),
                value(128, tag("align128")),
                value(256, tag("align256")),
                value(512, tag("align512")),
                value(16, tag("align16")),
                value(32, tag("align32")),
                value(64, tag("align64")),
                value(0, tag("align0")),
                value(1, tag("align1")),
                value(2, tag("align2")),
                value(4, tag("align4")),
                value(8, tag("align8")),
            )),
            space1,
            alt((
                map(
                    |i| literal(InstrSize::I64, i),
                    |lit| lit.as_integer::<u64>().unwrap().to_bytes().to_vec(),
                ),
                map(
                    tuple((
                        tag("resb"),
                        space1,
                        alt((
                            |i| decimal(InstrSize::I64, i),
                            |i| hexadecimal(InstrSize::I64, i),
                        )),
                    )),
                    |(_, _, tok)| vec![0x00; tok.as_integer::<u64>().unwrap() as usize],
                ),
                map(
                    preceded(
                        tag("\""),
                        terminated(
                            many1(alt((
                                alphanumeric1,
                                space1,
                                tag("!"),
                                recognize(tuple((tag("\\x"), hex_digit1))),
                            ))),
                            tag("\""),
                        ),
                    ),
                    |string| {
                        let mut data = Vec::new();
                        let bytes = string.join("").into_bytes();
                        let mut cursor = 0;
                        while cursor < bytes.len() {
                            if &bytes[cursor..bytes.len().min(cursor + 2)] == br"\x" {
                                let d = u8::from_str_radix(
                                    std::str::from_utf8(&bytes[cursor + 2..cursor + 4]).unwrap(),
                                    16,
                                )
                                .unwrap();
                                data.push(d);
                                cursor += 4;
                            } else {
                                data.push(bytes[cursor]);
                                cursor += 1;
                            }
                        }
                        data.push(0); // doesn't hurt
                        data
                    },
                ),
            )),
        )),
        |(_, name, _, align, _, data)| {
            Token::Data(Data {
                label: Label::new(name.join("")),
                data,
                align,
            })
        },
    )(i)
}

pub fn reg_offset(i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("["),
            opt(tag("-")),
            |i| decimal(InstrSize::I64, i),
            tag("+"),
            register,
            tag("]"),
        )),
        |(_, neg, off, _, reg, _)| {
            if let Token::I64(off) = off {
                if let Token::Register(reg) = reg {
                    if neg.is_some() {
                        Token::RegOffset(-(off as i64), reg)
                    } else {
                        Token::RegOffset(off as i64, reg)
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
    )(i)
}

pub fn lab_offset(i: &str) -> IResult<&str, Token> {
    map(
        tuple((
            tag("["),
            opt(tag("-")),
            |i| decimal(InstrSize::I64, i),
            tag("+"),
            data_tag,
            tag("]"),
        )),
        |(_, neg, off, _, lab, _)| {
            if let Token::I64(off) = off {
                if let Token::Data(dat) = lab {
                    if neg.is_some() {
                        Token::LabelOffset(-(off as i64), dat.label)
                    } else {
                        Token::LabelOffset(off as i64, dat.label)
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        },
    )(i)
}

pub fn val(size: InstrSize, i: &str) -> IResult<&str, Token> {
    alt((register, |i| literal(size, i), label, data_tag))(i)
}

pub fn addr(i: &str) -> IResult<&str, Token> {
    map(
        tuple((tag("["), |a| val(InstrSize::I64, a), tag("]"))),
        |res| Token::Addr(Box::new(res.1)),
    )(i)
}

pub fn token(size: InstrSize, asm: &str) -> IResult<&str, Token> {
    alt((|a| val(size, a), addr, reg_offset, lab_offset))(asm)
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
        )),
        alt((
            |asm| Opcode::Jmp.parse_asm(asm),
            |asm| Opcode::Jgt.parse_asm(asm),
            |asm| Opcode::Jlt.parse_asm(asm),
            |asm| Opcode::Jeq.parse_asm(asm),
            |asm| Opcode::Jne.parse_asm(asm),
            |asm| Opcode::Junoeq.parse_asm(asm),
            |asm| Opcode::Junone.parse_asm(asm),
            |asm| Opcode::Junolt.parse_asm(asm),
            |asm| Opcode::Junogt.parse_asm(asm),
            |asm| Opcode::Junole.parse_asm(asm),
            |asm| Opcode::Junoge.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Juno.parse_asm(asm),
            |asm| Opcode::Jordeq.parse_asm(asm),
            |asm| Opcode::Jordne.parse_asm(asm),
            |asm| Opcode::Jordlt.parse_asm(asm),
            |asm| Opcode::Jordgt.parse_asm(asm),
            |asm| Opcode::Jordle.parse_asm(asm),
            |asm| Opcode::Jordge.parse_asm(asm),
            |asm| Opcode::Jord.parse_asm(asm),
            |asm| Opcode::Call.parse_asm(asm),
            |asm| Opcode::Ret.parse_asm(asm),
        )),
        alt((
            |asm| Opcode::Shl.parse_asm(asm),
            |asm| Opcode::Shr.parse_asm(asm),
            |asm| Opcode::Sshr.parse_asm(asm),
            |asm| Opcode::Sext.parse_asm(asm),
            |asm| Opcode::Jge.parse_asm(asm),
            |asm| Opcode::Jle.parse_asm(asm),
            |asm| Opcode::Enpt.parse_asm(asm),
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
    pub fn assemble(
        &mut self,
        pc: u64,
        region_id: usize,
        line: &mut Vec<u8>,
    ) -> Option<UnlinkedRef> {
        match self {
            Token::I8(val) => line.extend_from_slice(&[LITERAL, *val]),
            Token::I16(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::I32(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::I64(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::I128(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::F32(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::F64(val) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&val.to_bytes());
            }
            Token::Register(reg) => {
                line.extend_from_slice(&[reg.mc_repr()]);
            }
            Token::RegOffset(off, reg) => {
                line.extend_from_slice(&[REGISTER_OFFSET]);
                line.extend_from_slice(&(*off as u64).to_bytes());
                line.extend_from_slice(&[reg.mc_repr()]);
            }
            Token::LabelOffset(off, lab) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&[0x00; 8]);
                return Some(UnlinkedRef {
                    ty: UnlinkedRefType::LabelOffset(*off),
                    label: lab.to_owned(),
                    region_id,
                    loc: pc + 1,
                });
            }
            Token::Label(lab) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&[0x00; 8]);
                return Some(UnlinkedRef {
                    ty: UnlinkedRefType::Label,
                    label: lab.to_owned(),
                    region_id,
                    loc: pc + 1,
                });
            }
            Token::Addr(adr) => {
                line.extend_from_slice(&[ADDRESS]);
                return adr.assemble(pc + 1, region_id, line);
            }
            Token::Unknown => panic!("Attempt to assemble an unknown token"), // todo: Err instead of panic
            Token::Data(data) => {
                line.extend_from_slice(&[LITERAL]);
                line.extend_from_slice(&[0x00; 8]);
                return Some(UnlinkedRef {
                    ty: UnlinkedRefType::Label,
                    label: data.label.to_owned(),
                    region_id,
                    loc: pc + 1,
                });
            }
        }
        None
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum UnlinkedRefType {
    Label,            // single-pointers to locations in memory
    LabelOffset(i64), // double-pointers that must be dereferenced to single-pointers at compile time
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct UnlinkedRef {
    pub ty: UnlinkedRefType,
    pub label: Label,
    pub region_id: usize,
    pub loc: u64,
}

impl Instr {
    pub const fn new(
        opcode: Opcode,
        size: InstrSize,
        arg0: Option<Token>,
        arg1: Option<Token>,
    ) -> Self {
        Self {
            opcode,
            size,
            arg0,
            arg1,
        }
    }

    pub fn assemble(
        &mut self,
        mut pc: u64,
        region_id: usize,
        line: &mut Vec<u8>,
    ) -> Result<(usize, Vec<UnlinkedRef>)> {
        let opcode = self.opcode.mc_repr();
        let start = pc;
        let size = self.size.mc_repr();
        line.extend_from_slice(&[opcode, size]);
        pc += 2;
        let mut refs = Vec::new();
        if let Some(ref mut arg) = self.arg0 {
            if let Some(r) = arg.assemble(pc, region_id, line) {
                refs.push(r);
            }
            pc += arg.mc_size_in_bytes() as u64;
        }
        if let Some(ref mut arg) = self.arg1 {
            if let Some(r) = arg.assemble(pc, region_id, line) {
                refs.push(r);
            }
            pc += arg.mc_size_in_bytes() as u64;
        }
        let size = pc - start;
        assert_eq!(size as usize, self.mc_size_in_bytes(), "{:?}", self);

        Ok((self.mc_size_in_bytes(), refs))
    }
}
