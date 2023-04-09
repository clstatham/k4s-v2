use std::cmp::Ordering;

use anyhow::{Context, Error, Result};
use rustc_hash::FxHashMap;

use crate::k4s::{
    parsers::machine::{parse_debug_symbols, tags},
    Instr, InstrSig, InstrSize, Opcode, Primitive, Register, Token,
};

bitflags::bitflags! {
    #[derive(Debug, Clone, Default)]
    pub struct Fl: u64 {
        const EQ = 0b1;
        const GT = 0b10;
        const ORD = 0b100;
    }
}

impl Fl {
    pub fn eq(&self) -> bool {
        self.intersects(Self::EQ)
    }
    pub fn gt(&self) -> bool {
        self.intersects(Self::GT)
    }
    pub fn lt(&self) -> bool {
        !self.intersects(Self::GT) && !self.intersects(Self::EQ)
    }
    pub fn ord(&self) -> bool {
        self.intersects(Self::ORD)
    }

    pub fn cmp(&mut self, a: &Token, b: &Token) {
        match a.partial_cmp(b) {
            Some(Ordering::Equal) => {
                self.insert(Self::ORD | Self::EQ);
            }
            Some(Ordering::Greater) => {
                self.insert(Self::ORD | Self::GT);
                self.remove(Self::EQ);
            }
            Some(Ordering::Less) => {
                self.insert(Self::ORD);
                self.remove(Self::GT | Self::EQ);
            }
            None => {
                self.remove(Self::ORD);
                self.remove(Self::GT);
                self.remove(Self::EQ);
            }
        }
    }

    pub fn scmp(&mut self, a: &Token, b: &Token) {
        match a.scmp(b) {
            Some(Ordering::Equal) => {
                self.insert(Self::ORD | Self::EQ);
            }
            Some(Ordering::Greater) => {
                self.insert(Self::ORD | Self::GT);
                self.remove(Self::EQ);
            }
            Some(Ordering::Less) => {
                self.insert(Self::ORD);
                self.remove(Self::GT | Self::EQ);
            }
            None => {
                self.remove(Self::ORD);
                self.remove(Self::GT);
                self.remove(Self::EQ);
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
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
    pub fn get(&self, reg: Register, size: InstrSize) -> Option<Token> {
        match reg {
            Register::Rz => Token::from_integer_size(0u8, size),
            Register::Ra => Token::from_integer_size(self.ra, size),
            Register::Rb => Token::from_integer_size(self.rb, size),
            Register::Rc => Token::from_integer_size(self.rc, size),
            Register::Rd => Token::from_integer_size(self.rd, size),
            Register::Re => Token::from_integer_size(self.re, size),
            Register::Rf => Token::from_integer_size(self.rf, size),
            Register::Rg => Token::from_integer_size(self.rg, size),
            Register::Rh => Token::from_integer_size(self.rh, size),
            Register::Ri => Token::from_integer_size(self.ri, size),
            Register::Rj => Token::from_integer_size(self.rj, size),
            Register::Rk => Token::from_integer_size(self.rk, size),
            Register::Rl => Token::from_integer_size(self.rl, size),
            Register::Bp => Token::from_integer_size(self.bp, size),
            Register::Sp => Token::from_integer_size(self.sp, size),
            Register::Pc => Token::from_integer_size(self.pc, size),
            Register::Fl => Token::from_integer_size(self.fl.bits(), size),
        }
    }

    pub fn set(&mut self, reg: Register, val: Token) {
        match reg {
            Register::Rz => panic!("Attempt to write to RZ register"),
            Register::Ra => self.ra = val.as_integer().unwrap(),
            Register::Rb => self.rb = val.as_integer().unwrap(),
            Register::Rc => self.rc = val.as_integer().unwrap(),
            Register::Rd => self.rd = val.as_integer().unwrap(),
            Register::Re => self.re = val.as_integer().unwrap(),
            Register::Rf => self.rf = val.as_integer().unwrap(),
            Register::Rg => self.rg = val.as_integer().unwrap(),
            Register::Rh => self.rh = val.as_integer().unwrap(),
            Register::Ri => self.ri = val.as_integer().unwrap(),
            Register::Rj => self.rj = val.as_integer().unwrap(),
            Register::Rk => self.rk = val.as_integer().unwrap(),
            Register::Rl => self.rl = val.as_integer().unwrap(),
            Register::Bp => self.bp = val.as_integer().unwrap(),
            Register::Sp => self.sp = val.as_integer().unwrap(),
            Register::Pc => self.pc = val.as_integer().unwrap(),
            Register::Fl => self.fl = Fl::from_bits_truncate(val.as_integer().unwrap()),
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
            InstrSize::I16 => Token::I16(u16::from_bytes(&self[addr..addr + 2]).unwrap()),
            InstrSize::I32 => Token::I32(u32::from_bytes(&self[addr..addr + 4]).unwrap()),
            InstrSize::F32 => Token::F32(f32::from_bytes(&self[addr..addr + 4]).unwrap()),
            InstrSize::I64 => Token::I64(u64::from_bytes(&self[addr..addr + 8]).unwrap()),
            InstrSize::F64 => Token::F64(f64::from_bytes(&self[addr..addr + 8]).unwrap()),
            InstrSize::I128 => Token::I128(u128::from_bytes(&self[addr..addr + 16]).unwrap()),
        }
    }

    fn poke(&mut self, t: &Token, addr: u64) {
        let addr = addr as usize;
        match t {
            Token::I8(v) => self[addr] = *v,
            Token::I16(v) => self[addr..addr + 2].copy_from_slice(&v.to_bytes()),
            Token::I32(v) => self[addr..addr + 4].copy_from_slice(&v.to_bytes()),
            Token::F32(v) => self[addr..addr + 4].copy_from_slice(&v.to_bytes()),
            Token::I64(v) => self[addr..addr + 8].copy_from_slice(&v.to_bytes()),
            Token::F64(v) => self[addr..addr + 8].copy_from_slice(&v.to_bytes()),
            Token::I128(v) => self[addr..addr + 16].copy_from_slice(&v.to_bytes()),
            _ => panic!("Invalid token for writing: {:?}", t),
        }
    }
}

pub enum MachineState {
    Continue,
    ContDontUpdatePc, // used for jumps, calls, rets
    Halt,
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
            return Err(Error::msg(format!(
                "Invalid k4s header tag: {:?} (expected {:?})",
                &program[..tags::HEADER_MAGIC.len()],
                tags::HEADER_MAGIC
            )));
        }
        let program = &program[tags::HEADER_MAGIC.len()..];
        if &program[..tags::HEADER_ENTRY_POINT.len()] != tags::HEADER_ENTRY_POINT {
            return Err(Error::msg(format!(
                "Invalid k4s entry point tag: {:?} (expected {:?})",
                &program[..tags::HEADER_ENTRY_POINT.len()],
                tags::HEADER_ENTRY_POINT
            )));
        }
        let program = &program[tags::HEADER_ENTRY_POINT.len()..];
        let entry_point = u64::from_bytes(&program[..8]).unwrap();
        let program = &program[8..];

        let (program, debug_symbols) =
            parse_debug_symbols(program).map_err(|err| err.to_owned())?;

        if &program[..tags::HEADER_END.len()] != tags::HEADER_END {
            return Err(Error::msg("Invalid k4s header end tag"));
        }
        let program = &program[tags::HEADER_END.len()..];
        ram[entry_point as usize..entry_point as usize + program.len()].copy_from_slice(program);
        regs.pc = entry_point;
        regs.sp = ram.len() as u64;
        Ok(MachineContext {
            ram,
            regs,
            debug_symbols,
        })
    }

    pub fn step(&mut self) -> Result<MachineState> {
        let chunk = &self.ram[self.regs.pc as usize..self.regs.pc as usize + 64];
        let (_, instr) = Instr::disassemble_next(chunk)
            .map_err(|err| err.to_owned())
            .context(format!(
                "Error parsing instruction\nFirst 16 bytes:\n{:?}",
                &chunk[..16]
            ))?;
        #[cfg(debug_assertions)]
        {
            // dbg!(&instr);
            println!(
                "{:016x} --> {}",
                self.regs.pc,
                &instr.display_with_symbols(&self.debug_symbols)
            );
        }

        if instr.opcode == Opcode::Hlt {
            return Ok(MachineState::Halt);
        }

        match self.emulate_instr(&instr)? {
            MachineState::Continue => {
                self.regs.pc += instr.mc_size_in_bytes() as u64;
                Ok(MachineState::Continue)
            }
            MachineState::ContDontUpdatePc => {
                assert_ne!(self.regs.pc, 0, "jump to null address");
                Ok(MachineState::Continue)
            }
            MachineState::Halt => Ok(MachineState::Halt),
        }
    }

    pub fn run_until_hlt(&mut self) -> Result<()> {
        loop {
            if let MachineState::Halt = self.step()? {
                return Ok(());
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
            Token::Addr(Box::new(Token::I64(
                (self
                    .regs
                    .get(reg, InstrSize::I64)
                    .unwrap()
                    .as_integer::<u64>()
                    .unwrap() as i64
                    + offset) as u64,
            )))
        } else {
            token
        }
    }

    fn register_to_value(&self, token: Token, size: InstrSize) -> Token {
        if let Token::Register(reg) = token {
            let res = self.regs.get(reg, size);
            if res.is_none() {
                dbg!(token, size);
            }
            res.unwrap()
        } else {
            token
        }
    }

    fn addr_to_value(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        if let Token::Addr(ref tok) = token {
            let addr = self.eval_token(*tok.to_owned(), InstrSize::I64)?;
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
        let token = self.register_to_value(token, target_size);
        let token = self.offset_to_addr(token);
        // let token = self.addr_to_value(token, target_size)?;
        // the token should now be in integer (or floating point) form
        assert!(
            matches!(
                token,
                Token::F32(_)
                    | Token::F64(_)
                    | Token::I8(_)
                    | Token::I16(_)
                    | Token::I32(_)
                    | Token::I64(_)
                    | Token::I128(_)
            ),
            "{:?}",
            token
        );
        Ok(token)
    }

    fn read0(&self, token: Token, instr: &Instr) -> Result<Token> {
        match instr.signature() {
            InstrSig::AdrVal | InstrSig::AdrAdr | InstrSig::Adr => {
                if let Token::Addr(_) = &token {
                    let val = self.addr_to_value(token, instr.size)?;
                    Ok(val)
                } else {
                    Err(Error::msg(format!(
                        "Error parsing first argument: {:?}",
                        token
                    )))
                }
            }
            InstrSig::ValAdr | InstrSig::ValVal | InstrSig::Val => {
                self.eval_token(token, instr.size)
            }
            _ => Err(Error::msg(format!(
                "Error parsing first argument: {:?}",
                token
            ))),
        }
    }

    fn read1(&self, token: Token, instr: &Instr) -> Result<Token> {
        match instr.signature() {
            InstrSig::ValAdr | InstrSig::AdrAdr => {
                if let Token::Addr(_) = &token {
                    let val = self.addr_to_value(token, instr.size)?;
                    Ok(val)
                } else {
                    Err(Error::msg(format!(
                        "Error parsing first argument: {:?}",
                        token
                    )))
                }
            }
            InstrSig::AdrVal | InstrSig::ValVal => self.eval_token(token, instr.size),
            _ => Err(Error::msg(format!(
                "Error parsing first argument: {:?}",
                token
            ))),
        }
    }

    fn assign_lvalue_with(
        &mut self,
        lvalue: Token,
        instr: &Instr,
        f: impl FnOnce(&Token) -> Result<Token>,
    ) -> Result<()> {
        match instr.signature() {
            InstrSig::AdrAdr | InstrSig::AdrVal | InstrSig::Adr => match lvalue {
                Token::Addr(ref addr) => {
                    let addr = self
                        .eval_token(*addr.to_owned(), InstrSize::I64)?
                        .as_integer()
                        .unwrap();
                    let a = self.ram.peek(instr.size, addr);
                    let a = f(&a)?;
                    self.ram.poke(&a, addr);
                    Ok(())
                }
                _ => Err(Error::msg(format!(
                    "Error parsing first argument: {:?}",
                    lvalue
                ))),
            },
            InstrSig::ValVal | InstrSig::ValAdr | InstrSig::Val => match lvalue {
                Token::Register(reg) => {
                    let a = self.regs.get(reg, instr.size).unwrap();
                    let a = f(&a)?;
                    self.regs.set(reg, a);
                    Ok(())
                }
                _ => Err(Error::msg(format!(
                    "Error parsing first argument: {:?}",
                    lvalue
                ))),
            },
            _ => Err(Error::msg(format!(
                "Error parsing first argument: {:?}",
                lvalue
            ))),
        }
    }

    fn emulate_instr(&mut self, instr: &Instr) -> Result<MachineState> {
        // let arg0_val = instr.arg0().and_then(|arg| self.read0(arg, instr));
        let arg0 = instr.arg0();
        let arg1 = instr.arg1().and_then(|arg| self.read1(arg, instr));
        match instr.opcode {
            Opcode::Hlt => return Ok(MachineState::Halt),
            Opcode::Nop => {}
            Opcode::Und => panic!("Program entered explicit undefined behavior"),
            Opcode::Mov => {
                self.assign_lvalue_with(arg0?, instr, |_| arg1)?;
            }
            Opcode::Push => {
                self.push(self.read0(arg0?, instr)?);
            }
            Opcode::Pop => {
                let rvalue = self.pop(instr.size);
                self.assign_lvalue_with(arg0?, instr, |_| Ok(rvalue))?;
            }
            Opcode::Add => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.add(&arg1?))?;
            }
            Opcode::Sub => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.sub(&arg1?))?;
            }
            Opcode::Mul => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.mul(&arg1?))?;
            }
            Opcode::Div => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.div(&arg1?))?;
            }
            Opcode::Mod => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.rem(&arg1?))?;
            }
            Opcode::And => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.bitand(&arg1?))?;
            }
            Opcode::Or => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.bitor(&arg1?))?;
            }
            Opcode::Xor => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.bitxor(&arg1?))?;
            }
            Opcode::Shl => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.shl(&arg1?))?;
            }
            Opcode::Shr => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.shr(&arg1?))?;
            }
            Opcode::Sext => {
                self.assign_lvalue_with(arg0?, instr, |arg0| Ok(arg0.sext(instr.size).unwrap()))?
            }
            Opcode::Sshr => self.assign_lvalue_with(arg0?, instr, |arg0| arg0.sshr(&arg1?))?,
            Opcode::Sdiv => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.sdiv(&arg1?))?;
            }
            Opcode::Smod => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.smod(&arg1?))?;
            }
            Opcode::Printi => {
                println!(
                    "{}",
                    self.read0(arg0?, instr)?.as_integer::<u128>().unwrap()
                );
            }
            Opcode::Printc => {
                print!(
                    "{}",
                    std::str::from_utf8(&[self.read0(arg0?, instr)?.as_integer::<u8>().unwrap()])
                        .unwrap()
                );
            }
            Opcode::Call => {
                self.push(Token::I64(self.regs.pc + instr.mc_size_in_bytes() as u64));
                self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc);
            }
            Opcode::Ret => {
                self.regs.pc = self.pop(InstrSize::I64).as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc);
            }
            Opcode::Cmp => {
                self.regs.fl.cmp(&self.read0(arg0?, instr)?, &arg1?);
            }
            Opcode::Scmp => {
                self.regs.fl.scmp(&self.read0(arg0?, instr)?, &arg1?);
            }
            Opcode::Jmp => {
                self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                return Ok(MachineState::ContDontUpdatePc);
            }
            Opcode::Jeq => {
                if self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jne => {
                if !self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jgt => {
                if self.regs.fl.gt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jlt => {
                if self.regs.fl.lt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jge => {
                if self.regs.fl.gt() || self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jle => {
                if self.regs.fl.lt() || self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jord => {
                if self.regs.fl.ord() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Juno => {
                if !self.regs.fl.ord() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junoeq => {
                if !self.regs.fl.ord() || self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junone => {
                if !self.regs.fl.ord() || !self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junolt => {
                if !self.regs.fl.ord() || self.regs.fl.lt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junogt => {
                if !self.regs.fl.ord() || self.regs.fl.gt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junole => {
                if !self.regs.fl.ord() || self.regs.fl.lt() || self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Junoge => {
                if !self.regs.fl.ord() || self.regs.fl.gt() || self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordeq => {
                if self.regs.fl.ord() && self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordne => {
                if self.regs.fl.ord() && !self.regs.fl.eq() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordlt => {
                if self.regs.fl.ord() && self.regs.fl.lt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordgt => {
                if self.regs.fl.ord() && self.regs.fl.gt() {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordle => {
                if self.regs.fl.ord() && (self.regs.fl.lt() || self.regs.fl.eq()) {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
            Opcode::Jordge => {
                if self.regs.fl.ord() && (self.regs.fl.gt() || self.regs.fl.eq()) {
                    self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                    return Ok(MachineState::ContDontUpdatePc);
                }
            }
        }
        Ok(MachineState::Continue)
    }
}
