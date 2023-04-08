use anyhow::{Result, Error};

use crate::k4s::{Primitive, InstrSize, Token, Register, parsers::machine::tags, Instr, Opcode};

#[derive(Debug, Clone, Copy, Default)]
pub struct Regs {
    pub ra: u64,
    pub rb: u64,
    pub rc: u64,
    pub rd: u64,
    pub re: u64,
    pub rf: u64,
    pub rg: u64,
    pub rh: u64,
    pub ri: u64,
    pub rj: u64,
    pub rk: u64,
    pub rl: u64,
    pub bp: u64,
    pub sp: u64,
    pub pc: u64,
    pub fl: u64,
}

impl Regs {
    pub fn get(&self, reg: Register) -> u64 {
        match reg {
            Register::Rz => 0,
            Register::Ra => self.ra,
            Register::Rb => self.rb,
            Register::Rc => self.rc,
            Register::Rd => self.rd,
            Register::Re => self.re,
            Register::Rf => self.rf,
            Register::Rg => self.rg,
            Register::Rh => self.rh,
            Register::Ri => self.ri,
            Register::Rj => self.rj,
            Register::Rk => self.rk,
            Register::Rl => self.rl,
            Register::Bp => self.bp,
            Register::Sp => self.sp,
            Register::Pc => self.pc,
            Register::Fl => self.fl,
        }
    }

    pub fn set(&mut self, reg: Register, val: u64) {
        match reg {
            Register::Rz => panic!("Attempt to write to RZ register"),
            Register::Ra => self.ra = val,
            Register::Rb => self.rb = val,
            Register::Rc => self.rc = val,
            Register::Rd => self.rd = val,
            Register::Re => self.re = val,
            Register::Rf => self.rf = val,
            Register::Rg => self.rg = val,
            Register::Rh => self.rh = val,
            Register::Ri => self.ri = val,
            Register::Rj => self.rj = val,
            Register::Rk => self.rk = val,
            Register::Rl => self.rl = val,
            Register::Bp => self.bp = val,
            Register::Sp => self.sp = val,
            Register::Pc => self.pc = val,
            Register::Fl => self.fl = val,
        }
    }
}

pub trait Ram {
    fn peek(&self, size: InstrSize, addr: u64) -> Token;
    fn poke(&mut self, t: &Token, addr: u64);
}

impl Ram for Box<[u8]> {
    fn peek(&self, size: InstrSize, addr: u64) -> Token {
        let addr = addr as usize;
        match size {
            InstrSize::Unsized => panic!("Attempt to read a size of zero"),
            InstrSize::I8 => Token::I8(self[addr]),
            InstrSize::I16 => Token::I16(u16::from_bytes(&self[addr..addr+2]).unwrap()),
            InstrSize::I32 => Token::I32(u32::from_bytes(&self[addr..addr+4]).unwrap()),
            InstrSize::F32 => Token::F32(f32::from_bytes(&self[addr..addr+4]).unwrap()),
            InstrSize::I64 => Token::I64(u64::from_bytes(&self[addr..addr+8]).unwrap()),
            InstrSize::F64 => Token::F64(f64::from_bytes(&self[addr..addr+8]).unwrap()),
            InstrSize::I128 => Token::I128(u128::from_bytes(&self[addr..addr+16]).unwrap()),
        }
    }

    fn poke(&mut self, t: &Token, addr: u64) {
        let addr = addr as usize;
        match t {
            Token::I8(v) => self[addr] = *v,
            Token::I16(v) => self[addr..addr+2].copy_from_slice(&v.to_bytes()),
            Token::I32(v) => self[addr..addr+4].copy_from_slice(&v.to_bytes()),
            Token::F32(v) => self[addr..addr+4].copy_from_slice(&v.to_bytes()),
            Token::I64(v) => self[addr..addr+8].copy_from_slice(&v.to_bytes()),
            Token::F64(v) => self[addr..addr+8].copy_from_slice(&v.to_bytes()),
            Token::I128(v) => self[addr..addr+16].copy_from_slice(&v.to_bytes()),
            _ => panic!("Invalid token for writing: {:?}", t)
        }
    }
}

pub struct MachineContext {
    pub ram: Box<[u8]>,
    pub regs: Regs,
}

impl MachineContext {
    pub fn new(program: &[u8], mem_size: usize) -> Result<MachineContext> {
        let mut ram = vec![0u8; mem_size].into_boxed_slice();
        let mut regs = Regs::default();
        if &program[..tags::HEADER_MAGIC.len()] != tags::HEADER_MAGIC {
            return Err(Error::msg(format!("Invalid k4s header tag: {:?} (expected {:?})", &program[..tags::HEADER_MAGIC.len()], tags::HEADER_MAGIC)));
        }
        let program = &program[tags::HEADER_MAGIC.len()..];
        if &program[..tags::HEADER_ENTRY_POINT.len()] != tags::HEADER_ENTRY_POINT {
            return Err(Error::msg(format!("Invalid k4s entry point tag: {:?} (expected {:?})", &program[..tags::HEADER_ENTRY_POINT.len()], tags::HEADER_ENTRY_POINT)));
        }
        let program = &program[tags::HEADER_ENTRY_POINT.len()..];
        let entry_point = u64::from_bytes(&program[..8]).unwrap();
        let program = &program[8..];
        if &program[..tags::HEADER_END.len()] != tags::HEADER_END {
            return Err(Error::msg("Invalid k4s header end tag"));
        }
        let program = &program[tags::HEADER_END.len()..];
        ram[entry_point as usize .. entry_point as usize + program.len()].copy_from_slice(program);
        regs.pc = entry_point;
        regs.sp = ram.len() as u64;
        Ok(MachineContext { ram, regs })
    }

    pub fn step(&mut self) -> Result<bool> {
        let chunk = &self.ram[self.regs.pc as usize .. self.regs.pc as usize + 64];
        let (_, instr) = Instr::disassemble_next(chunk).map_err(|err| err.to_owned())?;
        dbg!(&instr);
        if instr.opcode == Opcode::Hlt {
            return Ok(false)
        }
        self.regs.pc += instr.mc_size_in_bytes() as u64;
        Ok(true)
    }

    pub fn run_until_hlt(&mut self) -> Result<()> {
        while self.step()? {}
        Ok(())
    }
}
