use std::cmp::Ordering;

use anyhow::{Result, Error};
use rustc_hash::FxHashMap;

use crate::k4s::{Primitive, InstrSize, Token, Register, parsers::machine::{tags, parse_debug_symbols}, Instr, Opcode};

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Fl: u64 {
        const EQ = 1 << 0;
        const GT = 1 << 1;
        const ORD = 1 << 2;
    }
}

impl Fl {
    pub fn eq(self) -> bool {
        self.contains(Self::EQ)
    }
    pub fn gt(self) -> bool {
        self.contains(Self::GT)
    }
    pub fn lt(self) -> bool {
        !self.contains(Self::GT) && !self.contains(Self::EQ)
    }
    pub fn ord(self) -> bool {
        self.contains(Self::ORD)
    }

    pub fn cmp(&mut self, a: Token, b: Token) {
        match a.partial_cmp(&b) {
            Some(Ordering::Equal) => {
                self.insert(Self::ORD);
                self.remove(Self::GT);
                self.insert(Self::EQ);
            }
            Some(Ordering::Greater) => {
                self.insert(Self::ORD);
                self.remove(Self::EQ);
                self.insert(Self::GT);
            }
            Some(Ordering::Less) => {
                self.insert(Self::ORD);
                self.remove(Self::GT);
                self.remove(Self::EQ);
            }
            None => {
                self.remove(Self::ORD);
            }
        }
    }
}

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
    pub fl: Fl,
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
            Register::Fl => self.fl.bits(),
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
            Register::Fl => self.fl = Fl::from_bits_truncate(val),
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

pub enum MachineState {
    Continue,
    ContDontUpdatePc, // used for jumps, calls, rets
    Halt
}

pub struct MachineContext {
    pub ram: Box<[u8]>,
    pub regs: Regs,
    pub debug_symbols: FxHashMap<u64, String>,
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

        let (program, debug_symbols) = parse_debug_symbols(program).map_err(|err| err.to_owned())?;

        if &program[..tags::HEADER_END.len()] != tags::HEADER_END {
            return Err(Error::msg("Invalid k4s header end tag"));
        }
        let program = &program[tags::HEADER_END.len()..];
        ram[entry_point as usize .. entry_point as usize + program.len()].copy_from_slice(program);
        regs.pc = entry_point;
        regs.sp = ram.len() as u64;
        Ok(MachineContext { ram, regs, debug_symbols })
    }

    pub fn step(&mut self) -> Result<MachineState> {
        let chunk = &self.ram[self.regs.pc as usize .. self.regs.pc as usize + 64];
        let (_, instr) = Instr::disassemble_next(chunk).map_err(|err| err.to_owned())?;
        #[cfg(debug_assertions)]
        println!("{:016x} --> {}", self.regs.pc, &instr.display_with_symbols(&self.debug_symbols));
        if instr.opcode == Opcode::Hlt {
            return Ok(MachineState::Halt)
        }

        match self.emulate_instr(&instr)? {
            MachineState::Continue => {
                self.regs.pc += instr.mc_size_in_bytes() as u64;
                Ok(MachineState::Continue)
            }
            MachineState::ContDontUpdatePc => {
                Ok(MachineState::Continue)
            }
            MachineState::Halt => Ok(MachineState::Halt)
        }
    }

    pub fn run_until_hlt(&mut self) -> Result<()> {
        loop {
            if let MachineState::Halt = self.step()? {
                return Ok(())
            }
        }
    }

    fn push(&mut self, val: Token) {
        self.regs.sp -= val.value_size_in_bytes() as u64;
        self.ram.poke(&val, self.regs.sp);
    }

    fn pop(&mut self, size: InstrSize) -> Token {
        let val = self.ram.peek(size, self.regs.sp);
        self.regs.sp += val.value_size_in_bytes() as u64;
        val
    }

    fn offset_to_addr(&self, token: Token) -> Token {
        if let Token::Offset(offset, reg) = token {
            Token::Addr(Box::new(Token::I64((self.regs.get(reg) as i64 + offset) as u64)))
        } else {
            token
        }
    }

    fn register_to_value(&self, token: Token) -> Token {
        if let Token::Register(reg) = token {
            Token::I64(self.regs.get(reg))
        } else {
            token
        }
    }

    fn addr_to_value(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        if let Token::Addr(ref tok) = token {
            let addr = self.eval_token(*tok.to_owned(), target_size)?;
            if let Token::I64(addr) = addr {
                Ok(self.ram.peek(target_size, addr))
            } else {
                Err(Error::msg(format!("Error parsing addr token: {:?}", token)))
            }
        } else {
            Ok(token)
        }
    }

    fn eval_token(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        let token = self.register_to_value(token);
        let token = self.offset_to_addr(token);
        let token = self.addr_to_value(token, target_size)?;
        // the token should now be in integer (or floating point) form
        Ok(token)
    }

    fn assign_with_token(&mut self, lvalue: Token, rvalue: Token, target_size: InstrSize) -> Result<()> {
        match lvalue {
            Token::Register(reg) => {
                self.regs.set(reg, self.eval_token(rvalue, InstrSize::I64)?.as_integer().unwrap());
            }
            Token::Offset(_, _) => {
                let addr = self.offset_to_addr(lvalue);
                let addr = self.eval_token(addr, target_size)?;
                self.ram.poke(&self.eval_token(rvalue, target_size)?, addr.as_integer().unwrap());
            }
            Token::Addr(addr) => {
                let addr = self.eval_token(*addr, target_size)?;
                self.ram.poke(&self.eval_token(rvalue, target_size)?, addr.as_integer().unwrap());
            }
            _ => return Err(Error::msg(format!("Invalid lvalue token for assignment: {:?}", lvalue)))
        }
        Ok(())
    }


    fn emulate_instr(&mut self, instr: &Instr) -> Result<MachineState> {
        let arg0_val = instr.arg0().and_then(|arg| self.eval_token(arg, instr.size));
        let arg0 = instr.arg0();
        let arg1 = instr.arg1().and_then(|arg| self.eval_token(arg, instr.size));
        match instr.opcode {
            Opcode::Hlt => return Ok(MachineState::Halt),
            Opcode::Nop => {},
            Opcode::Und => panic!("Program entered explicit undefined behavior"),
            Opcode::Mov => {
                self.assign_with_token(arg0?, arg1?, instr.size)?;
            }
            Opcode::Push => {
                self.push(arg0_val?);
            }
            Opcode::Pop => {
                let rvalue = self.pop(instr.size);
                self.assign_with_token(arg0?, rvalue, instr.size)?;
            }
            Opcode::Add => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.add(&arg1?)?, instr.size)?;
            }
            Opcode::Sub => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.sub(&arg1?)?, instr.size)?;
            }
            Opcode::Mul => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.mul(&arg1?)?, instr.size)?;
            }
            Opcode::Div => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.div(&arg1?)?, instr.size)?;
            }
            Opcode::Mod => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.rem(&arg1?)?, instr.size)?;
            }
            Opcode::And => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.bitand(&arg1?)?, instr.size)?;
            }
            Opcode::Or => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.bitor(&arg1?)?, instr.size)?;
            }
            Opcode::Xor => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.bitxor(&arg1?)?, instr.size)?;
            }
            Opcode::Shl => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.shl(&arg1?)?, instr.size)?;
            }
            Opcode::Shr => {
                let arg0 = arg0?;
                self.assign_with_token(arg0.clone(), arg0.shr(&arg1?)?, instr.size)?;
            }
            Opcode::Printi => {
                println!("{}", arg0_val?.as_integer::<u128>().unwrap());
            }
            Opcode::Printc => {
                print!("{}", std::str::from_utf8(&[arg0_val?.as_integer::<u8>().unwrap()]).unwrap());
            }
            Opcode::Call => {
                self.push(Token::I64(self.regs.pc + instr.mc_size_in_bytes() as u64));
                self.regs.pc = arg0_val?.as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc)
            }
            Opcode::Ret => {
                self.regs.pc = self.pop(InstrSize::I64).as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc)
            }
            Opcode::Cmp | Opcode::Fcmp => { // todo: merge these
                self.regs.fl.cmp(arg0?, arg1?);
            }
            Opcode::Jmp => {
                self.regs.pc = arg0_val?.as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc)
            }
            Opcode::Jeq => {
                if self.regs.fl.eq() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jne => {
                if !self.regs.fl.eq() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jgt => {
                if self.regs.fl.gt() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jlt => {
                if self.regs.fl.lt() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jge => {
                if self.regs.fl.gt() || self.regs.fl.eq() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jle => {
                if self.regs.fl.lt() || self.regs.fl.eq() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Jord => {
                if self.regs.fl.ord() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            Opcode::Juno => {
                if !self.regs.fl.ord() {
                    self.regs.pc = arg0_val?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc)
                }
            }
            // Opcode::Sdiv => todo!(),
            // Opcode::Smod => todo!(),
            // Opcode::Scmp => todo!(),
            // Opcode::Junoeq => todo!(),
            // Opcode::Junone => todo!(),
            // Opcode::Junolt => todo!(),
            // Opcode::Junogt => todo!(),
            // Opcode::Junole => todo!(),
            // Opcode::Junoge => todo!(),
            // Opcode::Jordeq => todo!(),
            // Opcode::Jordne => todo!(),
            // Opcode::Jordlt => todo!(),
            // Opcode::Jordgt => todo!(),
            // Opcode::Jordle => todo!(),
            // Opcode::Jordge => todo!(),
            // Opcode::Sshr => todo!(),
            // Opcode::Sext => todo!(),
            
            _ => todo!("{:?}", instr)
        }
        Ok(MachineState::Continue)
    }
}
