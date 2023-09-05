use std::{cmp::Ordering, fmt::{Write, Display}, collections::{VecDeque, BTreeMap}};

use anyhow::{Context, Error, Result};


use crate::k4s::{
    parsers::machine::{parse_debug_symbols, tags},
    Instr, InstrSig, InstrSize, Opcode, Primitive, Register, Token,
};

pub mod ram;

bitflags::bitflags! {
    #[derive(Debug, Clone, Default)]
    pub struct Fl: u64 {
        const EQ = 1 << 0;
        const GT = 1 << 1;
        const ORD = 1 << 2;
        const RESERVED_0 = 1 << 3;
        const RESERVED_1 = 1 << 4;
        const PT_ENABLED = 1 << 5;
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
    pub fn pt_enabled(&self) -> bool {
        self.intersects(Self::PT_ENABLED)
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
    pub pt: u64,
}

impl Regs {
    pub fn get(&self, reg: Register, size: InstrSize) -> Result<Token> {
        match reg {
            Register::R0 => Token::from_integer_size(0u8, size),
            Register::R1 => Token::from_integer_size(1u8, size),
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
            Register::Pt => Token::from_integer_size(self.pt, size),
        }.ok_or(Error::msg(format!("Error interpreting {} as {}", reg, size)))
    }

    pub fn set(&mut self, reg: Register, val: Token) -> Result<()> {
        match reg {
            Register::R0 => return Err(Error::msg("Attempt to write to R0 register")),
            Register::R1 => return Err(Error::msg("Attempt to write to R1 register")),
            Register::Ra => self.ra = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rb => self.rb = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rc => self.rc = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rd => self.rd = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Re => self.re = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rf => self.rf = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rg => self.rg = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rh => self.rh = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Ri => self.ri = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rj => self.rj = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rk => self.rk = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Rl => self.rl = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Bp => self.bp = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Sp => self.sp = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Pc => self.pc = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
            Register::Fl => self.fl = Fl::from_bits_truncate(val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?),
            Register::Pt => self.pt = val.as_integer().ok_or(Error::msg(format!("Error interpreting {} as {}", val, InstrSize::I64)))?,
        }
        Ok(())
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
            "bp={:016x} sp={:016x} pc={:016x} fl={:016x} pt={:016x}",
            self.bp, self.sp, self.pc, self.fl, self.pt
        )
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
    pub debug_symbols: BTreeMap<u64, String>,
    pub instr_history: VecDeque<String>,
    pub output_history: String,
    pub stack_frame: Vec<(u64, u64)>,
    pub call_stack: Vec<String>,
    pub error: Option<String>,
    pub call_depth: usize,
}

impl MachineContext {
    pub fn new(programs: Vec<Vec<u8>>, mem_size: usize) -> Result<MachineContext> {
        let mut ram = vec![0x00; mem_size].into_boxed_slice();
        let mut regs = Regs::default();
        let mut first_entry_point = None;
        let mut all_debug_symbols = BTreeMap::new();
        for program in programs {
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
            if first_entry_point.is_none() {
                first_entry_point = Some(entry_point);
            }
            let program = &program[8..];
    
            let (program, debug_symbols) =
                parse_debug_symbols(program).map_err(|err| err.to_owned())?;
            for (adr, sym) in debug_symbols {
                all_debug_symbols.insert(adr, sym);
            }
    
            if &program[..tags::HEADER_END.len()] != tags::HEADER_END {
                return Err(Error::msg("Invalid k4s header end tag"));
            }
            let program = &program[tags::HEADER_END.len()..];
            ram[..program.len()].copy_from_slice(program);
        }
        
        // regs.bp = entry_point;
        regs.pc = first_entry_point.unwrap();

        regs.rg = mem_size as u64;

        Ok(MachineContext {
            ram,
            regs,
            debug_symbols: all_debug_symbols,
            instr_history: VecDeque::new(),
            call_stack: Vec::new(),
            stack_frame: Vec::new(),
            output_history: String::new(),
            error: None,
            call_depth: 0,
        })
    }

    pub fn step(&mut self, update_dbg_info: bool) -> Result<MachineState> {
        let pc = self.translate_addr(self.regs.pc)?;
        let chunk = self.peek_phys_range(pc, pc + 64)?;
        let (_, instr) = Instr::disassemble_next(chunk)
            .map_err(|err| err.to_owned())
            .context(format!(
                "Error parsing instruction\nPC={:x} (-> {:x})\nFirst 16 bytes:\n{:x?}\nor\n{:?}",
                self.regs.pc,
                pc,
                &chunk[..16],
                &chunk[..16]
            ))?;

        // let pc = self.regs.pc;
        // log::trace!("\n{}", self.regs);
        // if let Some(tok) = &instr.arg0 {
        //     let tok_eval = self
        //         .eval_token(tok.to_owned(), instr.size).unwrap_or(Token::Unknown);
        //     log::trace!("arg0 = {:?} ({:x?})", tok, tok_eval);
        // }
        // if let Some(tok) = &instr.arg1 {
        //     let tok_eval = self
        //         .eval_token(tok.to_owned(), instr.size).unwrap_or(Token::Unknown);
        //     log::trace!("arg1 = {:?} ({:x?})", tok, tok_eval);
        // }
        self.instr_history.push_back(format!("{}{}", "    ".repeat(self.call_depth), instr.display_with_symbols(&self.debug_symbols)));
        if self.instr_history.len() > 10000 {
            self.instr_history.pop_front();
        }
        if update_dbg_info {
            self.update_dbg_info()?;
        }
        
        log::trace!(
            "{:016x} --> {}",
            pc,
            &instr.display_with_symbols(&self.debug_symbols)
        );

        match self.emulate_instr(&instr).context(format!("Error emulating instruction `{}`", instr)) {
            Ok(MachineState::Continue) => {
                self.regs.pc += instr.mc_size_in_bytes() as u64;
                Ok(MachineState::Continue)
            }
            Ok(MachineState::ContDontUpdatePc) => {
                if self.regs.pc == 0 {
                    self.error = Some("Jump to null address".into());
                    Err(Error::msg("Jump to null address"))
                } else {
                    Ok(MachineState::Continue)
                }
            }
            Ok(MachineState::Halt) => Ok(MachineState::Halt),
            Err(e) => {
                self.error = Some(e.root_cause().to_string());
                Err(e)
            }
        }
    }

    pub fn find_symbol(&self, adr: u64) -> Option<String> {
        let mut it = self.debug_symbols.iter().peekable();
        let mut out_sym = None;
        while let Some((addr, sym)) = it.next() {
            if let Some((next_addr, _)) = it.peek() {
                if (addr..*next_addr).contains(&&adr) {
                    out_sym = Some(sym);
                    break;
                }
            } else if *addr <= adr {
                out_sym = Some(sym);
                break;
            }
        }
        out_sym.cloned()
    }

    pub fn update_dbg_info(&mut self) -> Result<()> {
        self.stack_frame = {
            let bp = self.regs.bp - self.regs.bp % 8;
            let sp = self.regs.sp - self.regs.sp % 8;
            let mut out = Vec::new();
            for adr in (sp..=bp).rev().step_by(8) {
                out.push((bp - adr, self.peek(InstrSize::I64, adr).unwrap_or(Token::I64(0)).as_integer().unwrap()));
            }
            out
        };
        self.call_stack = {
            let mut out = Vec::new();
            let mut bp = self.regs.bp;
            let mut depth = 16;
            while bp != 0 && depth > 0 {
                let rip_rbp = bp + 8;
                let rip = self.peek(InstrSize::I64, rip_rbp).unwrap_or(Token::I64(0)).as_integer::<u64>().unwrap();
                if rip == 0 {
                    break;
                }
                bp = self.peek(InstrSize::I64, bp).unwrap().as_integer::<u64>().unwrap();
                let sym = self.find_symbol(rip).unwrap_or("(unknown)".to_owned());
                out.push(sym);
                depth -= 1;
            }
            
            out
        };
        Ok(())
    }

    pub fn run_until_hlt(&mut self) -> Result<()> {
        loop {
            if let MachineState::Halt = self.step(false)? {
                return Ok(());
            }
        }
    }

    fn push(&mut self, val: Token) -> Result<()> {
        self.regs.sp -= val.value_size_in_bytes() as u64;
        self.poke(&val, self.regs.sp).context(format!("Error pushing `{}` to stack", val))
    }

    fn pop(&mut self, size: InstrSize) -> Result<Token> {
        let val = self.peek(size, self.regs.sp)?;
        self.regs.sp += val.value_size_in_bytes() as u64;
        Ok(val)
    }

    fn offset_to_addr(&self, token: Token) -> Token {
        if let Token::RegOffset(offset, reg) = token {
            Token::Addr(Box::new(Token::I64(
                (self
                    .regs
                    .get(reg, InstrSize::I64).unwrap() // safe since regs are always 64 bit
                    .as_integer::<u64>()
                    .unwrap() as i64 // safe since regs are always 64 bit
                    + offset) as u64,
            )))
        } else {
            token
        }
    }

    fn register_to_value(&self, token: Token, size: InstrSize) -> Result<Token> {
        if let Token::Register(reg) = token {
            self.regs.get(reg, size)
        } else {
            Ok(token)
        }
    }

    fn peek_addr(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        if let Token::Addr(ref tok) = token {
            if let Token::I64(addr) = **tok {
                self.peek(target_size, addr)
            } else if let Token::Register(reg) = **tok {
                self.peek(
                    target_size,
                    self.regs
                        .get(reg, InstrSize::I64).unwrap() // safe since regs are always 64 bit
                        .as_integer()
                        .unwrap(), // safe since regs are always 64 bit
                )
            } else {
                Err(Error::msg(format!("Error parsing addr token: `{}`", token)))
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
                Err(Error::msg(format!("Error parsing addr token: `{}`", token)))
            }
        } else {
            Ok(token)
        }
    }

    fn eval_token(&self, token: Token, target_size: InstrSize) -> Result<Token> {
        let token = self.register_to_value(token.to_owned(), target_size).context(format!("Error evaluating token `{}` to size `{}`", token, target_size))?;
        let token = self.offset_to_addr(token);
        let token = self.peek_addr(token.to_owned(), target_size).context(format!("Error evaluating token `{}` to size `{}`", token, target_size))?;
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
                        "Error parsing first argument: `{}`",
                        token
                    )))
                }
            }
            InstrSig::ValAdr | InstrSig::ValVal | InstrSig::Val => {
                self.eval_token(token, instr.size)
            }
            _ => Err(Error::msg(format!(
                "Error parsing first argument: `{}`",
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
                        Token::Register(reg) => self.regs.get(reg, InstrSize::I64).unwrap().as_integer().unwrap(),
                        _ => addr.as_integer().unwrap(),
                    };
                    let a = self.peek(instr.size, addr).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                    let a = f(&a)?;
                    self.poke(&a, addr).context(format!("Error assigning lvalue `{}` with `{}`", lvalue, a))?;
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
                    if instr.opcode == Opcode::Mov || instr.opcode == Opcode::Pop {
                        // mov and pop are special, since we don't actually do anything with whatever's currently in the register
                        // (and we could be putting a different InstrSize in there)
                        let a = f(&Token::Unknown).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                        self.regs.set(reg, a).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                        Ok(())
                    } else {
                        let a = self.regs.get(reg, instr.size).context(format!("Error interpreting register `{}` as `{}`", reg, instr.size))?;
                        let a = f(&a).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                        self.regs.set(reg, a).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                        Ok(())
                    }
                    
                }
                Token::RegOffset(..) => {
                    let addr = self.offset_to_addr(lvalue.to_owned());
                    let addr = self.addr_to_value(addr).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?.as_integer().unwrap();
                    // let value = self.addr_to_value(addr, instr.size).unwrap().as_integer().unwrap();
                    let a = self.peek(instr.size, addr).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                    let a = f(&a).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
                    self.poke(&a, addr).context(format!("Error assigning lvalue `{}` with the result of `{}`", lvalue, instr))?;
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
            Opcode::Und => return Err(Error::msg("Program entered explicit undefined behavior")),
            Opcode::Mov => {
                self.assign_lvalue_with(arg0?, instr, |_| arg1)?;
            }
            Opcode::Push => {
                self.push(self.read0(arg0?, instr)?)?;
            }
            Opcode::Pop => {
                let rvalue = self.pop(instr.size)?;
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

            Opcode::Sadd => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.sadd(&arg1?))?;
            }

            Opcode::Ssub => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.ssub(&arg1?))?;
            }
            Opcode::Smul => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.smul(&arg1?))?;
            }
            Opcode::Sdiv => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.sdiv(&arg1?))?;
            }
            Opcode::Smod => {
                self.assign_lvalue_with(arg0?, instr, |arg0| arg0.smod(&arg1?))?;
            }
            Opcode::Printi => {
                let val = self.read0(arg0?, instr)?;
                println!(
                    "{:x}",
                    val.as_integer::<u128>().unwrap()
                );
                writeln!(
                    &mut self.output_history,
                    "{:x}",
                    val.as_integer::<u128>().unwrap()
                )?;
            }
            Opcode::Printc => {
                let chr = self.read0(arg0?, instr)?;
                let chr = chr.as_integer::<u8>().ok_or(Error::msg(format!("Error converting token to u8: {:?}", chr)))?;
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
                self.push(Token::I64(self.regs.pc + instr.mc_size_in_bytes() as u64)).context("Error pushing return addr in `call` instruction")?;
                self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();
                self.call_depth += 1;
                return Ok(MachineState::ContDontUpdatePc);
            }
            Opcode::Ret => {
                self.regs.pc = self.pop(InstrSize::I64)?.as_integer().unwrap();
                self.call_depth -= 1;
                return Ok(MachineState::ContDontUpdatePc);
            }
            Opcode::Enpt => {
                self.regs.pc = self.read0(arg0?, instr)?.as_integer().unwrap();

                self.regs.fl.set(Fl::PT_ENABLED, true);
                self.call_depth -= 1;
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
