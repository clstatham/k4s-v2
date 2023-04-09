use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_until1},
    combinator::{map, recognize, value},
    multi::{many0},
    sequence::tuple,
    IResult,
};
use rustc_hash::FxHashMap;

use crate::k4s::{Instr, InstrSize, Opcode, Register, Token, Primitive};

use self::tags::{ADDRESS, LITERAL, REGISTER_OFFSET};

pub mod tags {
    pub const HEADER_MAGIC: &[u8] = b"k4d\x13\x37";
    pub const HEADER_ENTRY_POINT: &[u8] = b"ent";
    pub const HEADER_DEBUG_SYMBOLS_START: &[u8] = b"dbg";
    pub const HEADER_DEBUG_SYMBOLS_ENTRY_ADDR: &[u8] = b"db";
    pub const HEADER_DEBUG_SYMBOLS_ENTRY_END: &[u8] = b"\x00\x00";
    pub const HEADER_DEBUG_SYMBOLS_END: &[u8] = b"gbd";
    pub const HEADER_END: &[u8] = b"\x69\x69d4k";

    pub const LITERAL: u8 = 0xff;
    pub const REGISTER_OFFSET: u8 = 0xfe;
    pub const ADDRESS: u8 = 0xfd;
}

pub fn debug_entry(mc: &[u8]) -> IResult<&[u8], (String, u64)> {
    map(tuple((
        tag(tags::HEADER_DEBUG_SYMBOLS_ENTRY_ADDR),
        take(8_usize),
        take_until1(tags::HEADER_DEBUG_SYMBOLS_ENTRY_END),
        take(2_usize),
    )), |res: (&[u8], &[u8], &[u8], &[u8])| (String::from_utf8(res.2.to_vec()).unwrap(), u64::from_bytes(res.1).unwrap()))(mc)
}

pub fn parse_debug_symbols(mc: &[u8]) -> IResult<&[u8], FxHashMap<u64, String>> {
    map(
        tuple((
            tag(tags::HEADER_DEBUG_SYMBOLS_START),
            many0(debug_entry),
            tag(tags::HEADER_DEBUG_SYMBOLS_END),
        )),
        |(_, res, _)| res.into_iter().map(|(name, addr)| (addr, name)).collect(),
    )(mc)
}

impl InstrSize {
    pub fn parse_mc(self, mc: &[u8]) -> IResult<&[u8], InstrSize> {
        value(self, tag(&[self.mc_repr()]))(mc)
    }
}

pub fn parse_size(mc: &[u8]) -> IResult<&[u8], InstrSize> {
    alt((
        |mc| InstrSize::Unsized.parse_mc(mc),
        |mc| InstrSize::I8.parse_mc(mc),
        |mc| InstrSize::I16.parse_mc(mc),
        |mc| InstrSize::I32.parse_mc(mc),
        |mc| InstrSize::I64.parse_mc(mc),
        |mc| InstrSize::I128.parse_mc(mc),
        |mc| InstrSize::F32.parse_mc(mc),
        |mc| InstrSize::F64.parse_mc(mc),
    ))(mc)
}

pub fn parse_literal(size: InstrSize, mc: &[u8]) -> IResult<&[u8], Token> {
    match size {
        InstrSize::Unsized => Ok((mc, Token::Unknown)),
        InstrSize::I8 => map(tuple((tag(&[LITERAL]), take(1_usize))), |res| {
            Token::I8(u8::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::I16 => map(tuple((tag(&[LITERAL]), take(2_usize))), |res| {
            Token::I16(u16::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::I32 => map(tuple((tag(&[LITERAL]), take(4_usize))), |res| {
            Token::I32(u32::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::F32 => map(tuple((tag(&[LITERAL]), take(4_usize))), |res| {
            Token::F32(f32::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::I64 => map(tuple((tag(&[LITERAL]), take(8_usize))), |res| {
            Token::I64(u64::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::F64 => map(tuple((tag(&[LITERAL]), take(8_usize))), |res| {
            Token::F64(f64::from_bytes(res.1).unwrap())
        })(mc),
        InstrSize::I128 => map(tuple((tag(&[LITERAL]), take(16_usize))), |res| {
            Token::I128(u128::from_bytes(res.1).unwrap())
        })(mc),
    }
}

pub fn parse_register(mc: &[u8]) -> IResult<&[u8], Token> {
    // i could've done this using `Register::parse_mc`, but oh well, i already typed it all out
    map(
        alt((
            value(Register::Rz, |mc| Register::Rz.match_mc(mc)),
            value(Register::Ra, |mc| Register::Ra.match_mc(mc)),
            value(Register::Rb, |mc| Register::Rb.match_mc(mc)),
            value(Register::Rc, |mc| Register::Rc.match_mc(mc)),
            value(Register::Rd, |mc| Register::Rd.match_mc(mc)),
            value(Register::Re, |mc| Register::Re.match_mc(mc)),
            value(Register::Rf, |mc| Register::Rf.match_mc(mc)),
            value(Register::Rg, |mc| Register::Rg.match_mc(mc)),
            value(Register::Rh, |mc| Register::Rh.match_mc(mc)),
            value(Register::Ri, |mc| Register::Ri.match_mc(mc)),
            value(Register::Rj, |mc| Register::Rj.match_mc(mc)),
            value(Register::Rk, |mc| Register::Rk.match_mc(mc)),
            value(Register::Rl, |mc| Register::Rl.match_mc(mc)),
            value(Register::Bp, |mc| Register::Bp.match_mc(mc)),
            value(Register::Sp, |mc| Register::Sp.match_mc(mc)),
            value(Register::Pc, |mc| Register::Pc.match_mc(mc)),
            value(Register::Fl, |mc| Register::Fl.match_mc(mc)),
        )),
        Token::Register,
    )(mc)
}

pub fn parse_offset(mc: &[u8]) -> IResult<&[u8], Token> {
    map(
        tuple((tag(&[REGISTER_OFFSET]), take(8_usize), parse_register)),
        |res| {
            if let Token::Register(reg) = res.2 {
                Token::Offset(u64::from_bytes(res.1).unwrap() as i64, reg)
            } else {
                unreachable!()
            }
        },
    )(mc)
}

pub fn parse_addr(mc: &[u8]) -> IResult<&[u8], Token> {
    map(
        tuple((tag(&[ADDRESS]), |mc| disassemble_token(InstrSize::I64, mc))),
        |res| Token::Addr(Box::new(res.1)),
    )(mc)
}

impl Register {
    pub fn match_mc(self, mc: &[u8]) -> IResult<&[u8], &[u8]> {
        recognize(tag(&[self.mc_repr()]))(mc)
    }

    pub fn parse_mc(self, mc: &[u8]) -> IResult<&[u8], Register> {
        value(self, |mc| self.match_mc(mc))(mc)
    }
}

impl Opcode {
    pub fn parse_mc(self, mc: &[u8]) -> IResult<&[u8], Opcode> {
        value(self, |mc| recognize(tag(&[self.mc_repr()]))(mc))(mc)
    }
}

pub fn parse_opcode(mc: &[u8]) -> IResult<&[u8], Opcode> {
    alt((
        alt((
            |mc| Opcode::Und.parse_mc(mc),
            |mc| Opcode::Hlt.parse_mc(mc),
            |mc| Opcode::Mov.parse_mc(mc),
            |mc| Opcode::Push.parse_mc(mc),
            |mc| Opcode::Pop.parse_mc(mc),
            |mc| Opcode::Printi.parse_mc(mc),
            |mc| Opcode::Printc.parse_mc(mc),
            |mc| Opcode::Add.parse_mc(mc),
            |mc| Opcode::Sub.parse_mc(mc),
            |mc| Opcode::Mul.parse_mc(mc),
        )),
        alt((
            |mc| Opcode::Div.parse_mc(mc),
            |mc| Opcode::Sdiv.parse_mc(mc),
            |mc| Opcode::Mod.parse_mc(mc),
            |mc| Opcode::Smod.parse_mc(mc),
            |mc| Opcode::And.parse_mc(mc),
            |mc| Opcode::Or.parse_mc(mc),
            |mc| Opcode::Xor.parse_mc(mc),
            |mc| Opcode::Cmp.parse_mc(mc),
            |mc| Opcode::Scmp.parse_mc(mc),
            |mc| Opcode::Fcmp.parse_mc(mc),
        )),
        alt((
            |mc| Opcode::Jmp.parse_mc(mc),
            |mc| Opcode::Jgt.parse_mc(mc),
            |mc| Opcode::Jlt.parse_mc(mc),
            |mc| Opcode::Jeq.parse_mc(mc),
            |mc| Opcode::Jne.parse_mc(mc),
            |mc| Opcode::Juno.parse_mc(mc),
            |mc| Opcode::Junoeq.parse_mc(mc),
            |mc| Opcode::Junone.parse_mc(mc),
            |mc| Opcode::Junolt.parse_mc(mc),
            |mc| Opcode::Junogt.parse_mc(mc),
            |mc| Opcode::Junole.parse_mc(mc),
        )),
        alt((
            |mc| Opcode::Junoge.parse_mc(mc),
            |mc| Opcode::Jord.parse_mc(mc),
            |mc| Opcode::Jordeq.parse_mc(mc),
            |mc| Opcode::Jordne.parse_mc(mc),
            |mc| Opcode::Jordlt.parse_mc(mc),
            |mc| Opcode::Jordgt.parse_mc(mc),
            |mc| Opcode::Jordle.parse_mc(mc),
            |mc| Opcode::Jordge.parse_mc(mc),
            |mc| Opcode::Call.parse_mc(mc),
            |mc| Opcode::Ret.parse_mc(mc),
        )),
        alt((
            |mc| Opcode::Shl.parse_mc(mc),
            |mc| Opcode::Shr.parse_mc(mc),
            |mc| Opcode::Sshr.parse_mc(mc),
            |mc| Opcode::Sext.parse_mc(mc),
            |mc| Opcode::Jge.parse_mc(mc),
            |mc| Opcode::Jle.parse_mc(mc),
        )),
    ))(mc)
}


pub fn disassemble_token(size: InstrSize, mc: &[u8]) -> IResult<&[u8], Token> {
    alt((
        |mc| parse_literal(size, mc),
        parse_offset,
        parse_register,
        parse_addr,
    ))(mc)
}

impl Instr {
    pub fn disassemble_next(mc: &[u8]) -> IResult<&[u8], Instr> {
        let (mc, opcode) = take(1_usize)(mc)?;
        let opcode = parse_opcode(opcode)?.1;
        let (mc, size) = take(1_usize)(mc)?;
        let size = parse_size(size)?.1;

        let (arg0, arg1) = if opcode.n_args() > 0 {
            let arg0 = disassemble_token(size, mc)?;
            let arg1 = if opcode.n_args() > 1 {
                Some(disassemble_token(size, arg0.0)?.1)
            } else {
                None
            };
            (Some(arg0.1), arg1)
        } else {
            (None, None)
        };
        Ok((
            mc,
            Instr {
                opcode,
                size,
                arg0,
                arg1,
            },
        ))
    }
}
