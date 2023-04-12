use std::{cmp::Ordering, fmt::{Write, Display}};

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
    pub fn get(&self, reg: Register, size: InstrSize) -> Token {
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
        }.unwrap_or_else(|| panic!("Error interpreting {} as {}", reg, size))
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

impl Display for Regs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ra={:016x} rb={:016x} rc={:016x} rd={:016x} re={:016x} rf={:016x}",
            self.ra, self.rb, self.rc, self.rd, self.re, self.rf
        )?;
        writeln!(
            f,
            "rg={:016x} rh={:016x} ri={:016x} rj={:016x} rk={:016x} rl={:016x}",
            self.rg, self.rh, self.ri, self.rj, self.rk, self.rl
        )?;
        write!(
            f,
            "bp={:016x} sp={:016x} pc={:016x} fl={:016x}",
            self.bp, self.sp, self.pc, self.fl
        )
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
    pub instr_history: Vec<String>,
    pub output_history: String,
    pub stack_frame: Vec<(u64, u64)>,
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
            instr_history: Vec::new(),
            stack_frame: Vec::new(),
            output_history: String::new(),
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
        
        // dbg!(&instr);
        

        if instr.opcode == Opcode::Hlt {
            return Ok(MachineState::Halt);
        }

        let res = match self.emulate_instr(&instr)? {
            MachineState::Continue => {
                self.regs.pc += instr.mc_size_in_bytes() as u64;
                Ok(MachineState::Continue)
            }
            MachineState::ContDontUpdatePc => {
                assert_ne!(self.regs.pc, 0, "jump to null address");
                Ok(MachineState::Continue)
            }
            MachineState::Halt => Ok(MachineState::Halt),
        };
        match res {
            Ok(MachineState::Halt) => Ok(MachineState::Halt),
            Ok(_) => {
                log::debug!("\n{}", self.regs);
                if let Some(tok) = &instr.arg0 {
                    let tok_eval = self
                        .eval_token(tok.to_owned(), InstrSize::I64)?
                        .as_integer::<u64>()
                        .unwrap();
                    log::debug!("arg0 = {:?} ({:x?})", tok, tok_eval);
                }
                if let Some(tok) = &instr.arg1 {
                    let tok_eval = self
                        .eval_token(tok.to_owned(), InstrSize::I64)?
                        .as_integer::<u64>()
                        .unwrap();
                    log::debug!("arg1 = {:?} ({:x?})", tok, tok_eval);
                }
                log::debug!(
                    "{:016x} --> {}",
                    self.regs.pc,
                    &instr.display_with_symbols(&self.debug_symbols)
                );
                self.instr_history.push(format!("{:010x} ==> {}", self.regs.pc, instr.display_with_symbols(&self.debug_symbols)));
                self.stack_frame = {
                    let bp = self.regs.bp - self.regs.bp % 8;
                    let sp = self.regs.sp - self.regs.sp % 8;
                    let mut out = Vec::new();
                    for adr in (sp..=bp).rev().step_by(8) {
                        out.push((bp - adr, self.ram.peek(InstrSize::I64, adr).as_integer().unwrap()));
                    }
                    out
                };
                Ok(MachineState::Continue)
            }
            Err(e) => Err(e)
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
        if let Token::RegOffset(offset, reg) = token {
            Token::Addr(Box::new(Token::I64(
                (self
                    .regs
                    .get(reg, InstrSize::I64)
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
            self.regs.get(reg, size)
        } else {
            token
        }
    }

    fn peek_addr(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        if let Token::Addr(ref tok) = token {
            // let addr = self.eval_token(*tok.to_owned(), InstrSize::I64)?;
            if let Token::I64(addr) = **tok {
                Ok(self.ram.peek(target_size, addr))
            } else if let Token::Register(reg) = **tok {
                Ok(self.ram.peek(
                    target_size,
                    self.regs
                        .get(reg, InstrSize::I64)
                        .as_integer()
                        .unwrap(),
                ))
            } else {
                Err(Error::msg(format!("Error parsing addr token: {:?}", token)))
            }
        } else {
            Ok(token)
        }
    }

    fn addr_to_value(&self, token: Token) -> Result<Token> {
        if let Token::Addr(ref tok) = token {
            let addr = self.eval_token(*tok.to_owned(), InstrSize::I64)?;
            if let Token::I64(_) = addr {
                Ok(addr)
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
        let token = self.peek_addr(token, target_size)?;
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
                    let val = self.peek_addr(token, instr.size)?;
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

    fn assign_lvalue_with(
        &mut self,
        lvalue: Token,
        instr: &Instr,
        f: impl FnOnce(&Token) -> Result<Token>,
    ) -> Result<()> {
        
        match instr.signature() {
            InstrSig::AdrAdr | InstrSig::AdrVal | InstrSig::Adr => match lvalue {
                Token::Addr(ref addr) => {
                    let addr = match *addr.to_owned() {
                        Token::Register(reg) => self.regs.get(reg, InstrSize::I64).as_integer().unwrap(),
                        _ => addr.as_integer().unwrap(),
                    };
                    let a = self.ram.peek(instr.size, addr);
                    let a = f(&a)?;
                    self.ram.poke(&a, addr);
                    Ok(())
                }
                _ => Err(Error::msg(format!(
                    "Error parsing first argument: {:?}\nInvalid token for instruction signature {:?}",
                    lvalue,
                    instr.signature(),
                ))),
            },
            InstrSig::ValVal | InstrSig::ValAdr | InstrSig::Val => match lvalue {
                Token::Register(reg) => {
                    if instr.opcode == Opcode::Mov {
                        // mov is special, since we don't actually do anything with whatever's currently in the register
                        // (and we could be putting a different InstrSize in there)
                        let a = f(&Token::Unknown)?;
                        self.regs.set(reg, a);
                        Ok(())
                    } else {
                        let a = self.regs.get(reg, instr.size);
                        let a = f(&a)?;
                        self.regs.set(reg, a);
                        Ok(())
                    }
                    
                }
                Token::RegOffset(..) => {
                    let addr = self.offset_to_addr(lvalue);
                    let addr = self.addr_to_value(addr)?.as_integer().unwrap();
                    // let value = self.addr_to_value(addr, instr.size).unwrap().as_integer().unwrap();
                    let a = self.ram.peek(instr.size, addr);
                    let a = f(&a)?;
                    self.ram.poke(&a, addr);
                    Ok(())
                }
                _ => Err(Error::msg(format!(
                    "Error parsing first argument: {:?}\nInvalid token for instruction signature {:?}",
                    lvalue,
                    instr.signature(),
                ))),
            },
            _ => Err(Error::msg(format!(
                "Error parsing first argument: {:?}\nInvalid instruction signature {:?}",
                lvalue,
                instr.signature()
            ))),
        }
    }

    fn emulate_instr(&mut self, instr: &Instr) -> Result<MachineState> {
        let arg0 = instr.arg0();
        let arg1 = instr
            .arg1()
            .and_then(|arg| self.eval_token(arg, instr.size));
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
                let val = self.read0(arg0?, instr)?;
                println!(
                    "{}",
                    val.as_integer::<u128>().unwrap()
                );
                writeln!(
                    &mut self.output_history,
                    "{}",
                    val.as_integer::<u128>().unwrap()
                )?;
            }
            Opcode::Printc => {
                let chr = self.read0(arg0?, instr)?;
                let chr = chr.as_integer::<u8>().ok_or(Error::msg(format!("Error converting to u8: {:?}", chr)))?;
                print!(
                    "{}",
                    std::str::from_utf8(&[chr]).map_err(|_| Error::msg(format!("Error converting to utf-8: {:2x}", chr)))?
                );
                write!(
                    &mut self.output_history,
                    "{}",
                    std::str::from_utf8(&[chr]).map_err(|_| Error::msg(format!("Error converting to utf-8: {:2x}", chr)))?
                )?;
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
