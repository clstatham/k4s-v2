use anyhow::{Error, Result};

use crate::k4s::{InstrSize, Primitive, Token};

use super::MachineContext;

impl MachineContext {
    pub fn translate_addr(&self, addr: u64) -> Result<u64> {
        if self.regs.fl.pt_enabled() {
            let pt4 = self.regs.pt;
            if pt4 == 0 {
                return Err(Error::msg("PT register was enabled but null"));
            }
            if pt4 % 4096 != 0 {
                return Err(Error::msg("PT register was enabled but unaligned"));
            }
            let pt4_idx = ((addr / 4096) >> 27) & 0x1ff;
            let pt3_idx = ((addr / 4096) >> 18) & 0x1ff;
            let pt2_idx = ((addr / 4096) >> 9) & 0x1ff;
            let pt1_idx = (addr / 4096) & 0x1ff;
            let page_offset = addr & 0xfff;

            let pt3 = pt4 + pt4_idx * 8;
            let pt3 = u64::from_bytes(self.peek_phys_range(pt3, pt3 + 8)?).unwrap();
            if pt3 == 0 {
                return Err(Error::msg(format!(
                    "No level 3 page table entry for {:x}",
                    addr
                )));
            }
            if pt3 & 0b1 == 0 {
                return Err(Error::msg(format!(
                    "Level 3 page table entry for {:x} found, but wasn't marked present",
                    addr
                )));
            }
            let pt3 = pt3 & !0xfff;

            let pt2 = pt3 + pt3_idx * 8;
            let pt2 = u64::from_bytes(self.peek_phys_range(pt2, pt2 + 8)?).unwrap();
            if pt2 == 0 {
                return Err(Error::msg(format!(
                    "No level 2 page table entry for {:x}",
                    addr
                )));
            }
            if pt2 & 0b1 == 0 {
                return Err(Error::msg(format!(
                    "Level 2 page table entry for {:x} found, but wasn't marked present",
                    addr
                )));
            }
            let pt2 = pt2 & !0xfff;

            let pt1 = pt2 + pt2_idx * 8;
            let pt1 = u64::from_bytes(self.peek_phys_range(pt1, pt1 + 8)?).unwrap();
            if pt1 == 0 {
                return Err(Error::msg(format!(
                    "No level 1 page table entry for {:x}",
                    addr
                )));
            }
            if pt1 & 0b1 == 0 {
                return Err(Error::msg(format!(
                    "Level 1 page table entry for {:x} found, but wasn't marked present",
                    addr
                )));
            }
            let pt1 = pt1 & !0xfff;

            let frame = pt1 + pt1_idx * 8;
            let frame = u64::from_bytes(self.peek_phys_range(frame, frame + 8)?).unwrap();
            let frame = frame & !0xfff;
            let paddr = frame + page_offset;
            Ok(paddr)
        } else {
            Ok(addr)
        }
    }

    pub fn peek_phys_range(&self, lo: u64, hi: u64) -> Result<&[u8]> {
        self.ram
            .get(lo as usize..hi as usize)
            .ok_or(Error::msg("Out of bounds memory access"))
    }

    pub fn peek_phys_range_mut(&mut self, lo: u64, hi: u64) -> Result<&mut [u8]> {
        self.ram
            .get_mut(lo as usize..hi as usize)
            .ok_or(Error::msg("Out of bounds memory access"))
    }

    pub fn peek(&self, size: InstrSize, addr: u64) -> Result<Token> {
        let addr = self.translate_addr(addr)?;
        if addr == 0 {
            return Err(Error::msg(format!(
                "Attempt to read a size of {:?} at a null address",
                size
            )));
        }
        if addr as usize + size.in_bytes() > self.ram.len() {
            return Err(Error::msg(format!(
                "Attempt to read a size of {:?} at address past the end of memory: {:#x}",
                size, addr
            )));
        }
        match size {
            InstrSize::Unsized => Err(Error::msg(format!(
                "Attempt to read a size of zero at: {:#x}",
                addr
            ))),
            InstrSize::I8 => Ok(Token::I8(self.peek_phys_range(addr, addr + 1)?[0])),
            InstrSize::I16 => Ok(Token::I16(
                u16::from_bytes(self.peek_phys_range(addr, addr + 2)?).unwrap(),
            )),
            InstrSize::I32 => Ok(Token::I32(
                u32::from_bytes(self.peek_phys_range(addr, addr + 4)?).unwrap(),
            )),
            InstrSize::F32 => Ok(Token::F32(
                f32::from_bytes(self.peek_phys_range(addr, addr + 4)?).unwrap(),
            )),
            InstrSize::I64 => Ok(Token::I64(
                u64::from_bytes(self.peek_phys_range(addr, addr + 8)?).unwrap(),
            )),
            InstrSize::F64 => Ok(Token::F64(
                f64::from_bytes(self.peek_phys_range(addr, addr + 8)?).unwrap(),
            )),
            InstrSize::I128 => Ok(Token::I128(
                u128::from_bytes(self.peek_phys_range(addr, addr + 16)?).unwrap(),
            )),
        }
    }

    pub fn poke(&mut self, t: &Token, addr: u64) -> Result<()> {
        let addr = self.translate_addr(addr)?;
        if addr == 0 {
            return Err(Error::msg(format!(
                "Attempt to write {:?} at null address",
                t
            )));
        }
        if addr as usize + t.instr_size().in_bytes() > self.ram.len() {
            return Err(Error::msg(format!(
                "Attempt to write a size of {:?} at address past the end of memory: {:#x}",
                t.instr_size(),
                addr
            )));
        }
        match t {
            Token::I8(v) => self.peek_phys_range_mut(addr, addr + 1)?[0] = *v,
            Token::I16(v) => self
                .peek_phys_range_mut(addr, addr + 2)?
                .copy_from_slice(&v.to_bytes()),
            Token::I32(v) => self
                .peek_phys_range_mut(addr, addr + 4)?
                .copy_from_slice(&v.to_bytes()),
            Token::F32(v) => self
                .peek_phys_range_mut(addr, addr + 4)?
                .copy_from_slice(&v.to_bytes()),
            Token::I64(v) => self
                .peek_phys_range_mut(addr, addr + 8)?
                .copy_from_slice(&v.to_bytes()),
            Token::F64(v) => self
                .peek_phys_range_mut(addr, addr + 8)?
                .copy_from_slice(&v.to_bytes()),
            Token::I128(v) => self
                .peek_phys_range_mut(addr, addr + 16)?
                .copy_from_slice(&v.to_bytes()),
            _ => return Err(Error::msg(format!("Invalid token for writing: {:?}", t))),
        }
        Ok(())
    }
}
