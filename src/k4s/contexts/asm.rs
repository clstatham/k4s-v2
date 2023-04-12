use std::{fs::File, io::Read};

use anyhow::{Context, Error, Result};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::k4s::{
    parsers::{
        asm::{data, header, lab_offset_const, label, literal, opcode, size, token},
        machine::tags,
    },
    Data, Instr, InstrSize, Label, Linkage, Primitive, Token,
};

#[derive(Debug, Clone)]
pub enum Header {
    Entry(u64),
    Include(String),
}

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Comment,
    Header(Header),
    Label(Label),
    Data(Data),
    LabelOffsetConst(Label, i64, Label),
    Instr(Instr),
}

impl ParsedLine {
    pub fn parse(line: &str) -> anyhow::Result<(&str, Self)> {
        let line = line.trim();
        if line.starts_with(';') {
            return Ok((line, ParsedLine::Comment));
        }
        if let Ok((rest, header)) = header(line) {
            let rest = rest.trim();
            let header = match header {
                "!ent" => {
                    if rest.starts_with("0x") {
                        let rest = rest.strip_prefix("0x").unwrap_or(rest);
                        Header::Entry(u64::from_str_radix(rest, 16).unwrap())
                    } else {
                        return Err(Error::msg("Expected hex number after `!ent` tag"));
                    }
                }
                "!include" => {
                    let rest = rest.strip_prefix('\"').unwrap_or(rest);
                    let rest = rest.strip_suffix('\"').unwrap_or(rest);
                    Header::Include(rest.to_owned())
                }
                _ => return Err(Error::msg(format!("Invalid header tag: {header}"))),
            };
            Ok(("", ParsedLine::Header(header)))
        } else if let Ok((rest, (name, lab_off))) = lab_offset_const(line) {
            if let Token::LabelOffset(off, lab) = lab_off {
                Ok((rest, ParsedLine::LabelOffsetConst(name, off, lab)))
            } else {
                unreachable!()
            }
        } else if let Ok((rest, label)) = label(line) {
            if let Token::Label(label) = label {
                Ok((rest, ParsedLine::Label(label)))
            } else {
                unreachable!()
            }
        } else if let Ok((rest, data)) = data(line) {
            if let Token::Data(data) = data {
                Ok((rest, ParsedLine::Data(data)))
            } else {
                unreachable!()
            }
        } else {
            let (rest, opcode) = opcode(line)
                .map_err(|e| e.to_owned())
                .context("failed to parse opcode")?;
            if opcode.n_args() > 0 {
                let (rest, size) = size(rest.trim())
                    .map_err(|e| e.to_owned())
                    .context("failed to parse instruction size")?;
                let (rest, arg0, arg1) = if opcode.n_args() > 0 {
                    let (rest, arg0) = token(size, rest.trim())
                        .map_err(|e| e.to_owned())
                        .context("failed to parse first argument")?;
                    let (rest, arg1) = if opcode.n_args() > 1 {
                        let arg1 = token(size, rest.trim())
                            .map_err(|e| e.to_owned())
                            .context("failed to parse second argument")?;
                        (arg1.0, Some(arg1.1))
                    } else {
                        (rest, None)
                    };
                    (rest, Some(arg0), arg1)
                } else {
                    (rest, None, None)
                };
                Ok((
                    rest,
                    ParsedLine::Instr(Instr {
                        opcode,
                        size,
                        arg0,
                        arg1,
                    }),
                ))
            } else {
                Ok((
                    rest,
                    ParsedLine::Instr(Instr {
                        opcode,
                        size: InstrSize::Unsized,
                        arg0: None,
                        arg1: None,
                    }),
                ))
            }
        }
    }
}

pub struct AssemblyContext {
    pub lines: Vec<ParsedLine>,
    pub linked_refs: FxHashMap<Label, u64>, // (label, addr found at)
    pub unlinked_refs: FxHashMap<Label, FxHashSet<u64>>, // (label, addrs found at)
    pub linked_offset_refs: FxHashMap<(i64, Label), u64>, // ((offset, label), addr found at)
    pub unlinked_offset_refs: FxHashMap<(i64, Label), FxHashSet<u64>>, // ((offset, label), addrs found at)

    pub included_modules: FxHashSet<String>,
    pub input: String,
    pub pc: u64,
    pub entry_point: u64,
    pub output: Vec<u8>,
}

impl AssemblyContext {
    pub fn new(input: String) -> Self {
        Self {
            lines: Vec::default(),
            linked_refs: FxHashMap::default(),
            unlinked_refs: FxHashMap::default(),
            linked_offset_refs: FxHashMap::default(),
            unlinked_offset_refs: FxHashMap::default(),
            included_modules: FxHashSet::default(),
            input,
            pc: 0,
            entry_point: 0,
            output: Vec::new(),
        }
    }

    pub fn push_program_bytes(&mut self, bytes: &[u8]) {
        self.output.extend_from_slice(bytes);
        self.pc += bytes.len() as u64;
    }

    pub fn assemble(&mut self) -> Result<Vec<u8>> {
        self.assemble_impl(
            None,
            true,
            false,
            &FxHashSet::default(),
            &FxHashMap::default(),
        ) // todo: re-enable link checking
    }

    fn assemble_impl(
        &mut self,
        entry_point: Option<u64>,
        include_header: bool,
        check_link: bool,
        existing_includes: &FxHashSet<String>,
        existing_linked_refs: &FxHashMap<Label, u64>,
    ) -> Result<Vec<u8>> {
        let mut in_header = true;
        if let Some(entry_point) = entry_point {
            self.entry_point = entry_point;
            self.pc = entry_point;
        }

        self.linked_refs = existing_linked_refs.clone();

        for (line_no, line) in self.input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let line_no = line_no + 1;
            let (junk, mut parsed_line) = ParsedLine::parse(line)
                .context(format!("Error while parsing line {}:\n{}", line_no, line))?;
            if let ParsedLine::Data(ref mut data) = parsed_line {
                // parse the junk as data
                let string = junk.trim();
                if let Ok((_junk, lit)) = literal(InstrSize::I64, string) {
                    let lit = lit.as_integer::<u64>().unwrap();
                    data.data = lit.to_bytes().to_vec();
                } else if let Some(string) = string.strip_prefix('"') {
                    if let Some(string) = string.strip_suffix('"') {
                        let bytes = string.to_owned().into_bytes();
                        let mut cursor = 0;
                        while cursor < bytes.len() {
                            if &bytes[cursor..bytes.len().min(cursor + 2)] == br"\x" {
                                let d = u8::from_str_radix(
                                    std::str::from_utf8(&bytes[cursor + 2..cursor + 4]).unwrap(),
                                    16,
                                )
                                .unwrap();
                                data.data.push(d);
                                cursor += 4;
                            } else {
                                data.data.push(bytes[cursor]);
                                cursor += 1;
                            }
                        }
                        data.data.push(0);
                    } else {
                        return Err(Error::msg(format!(
                            "Expected closing `\"` after opening `\"` in data on line {line_no}"
                        )));
                    }
                } else {
                    return Err(Error::msg(format!("Invalid data on line {line_no}")));
                }
            } else if !junk.is_empty() && !junk.trim().starts_with(';') {
                log::warn!("Ignoring junk after line {line_no}: {junk}");
            }
            if let ParsedLine::Header(ref header) = parsed_line {
                if !in_header {
                    return Err(Error::msg(format!(
                        "Found header line outside of header on line {line_no}: {line}"
                    )));
                }

                match header {
                    Header::Entry(pc) => {
                        if entry_point.is_some() {
                            return Err(Error::msg(format!(
                                "Found entry point header line in non-main module on line {line_no}: {line}"
                            )));
                        }
                        self.pc = *pc;
                        self.entry_point = *pc;
                    }
                    Header::Include(path) => {
                        if !existing_includes
                            .union(&self.included_modules)
                            .collect::<FxHashSet<_>>()
                            .contains(&path.to_owned())
                        {
                            self.included_modules.insert(path.to_owned());
                        }
                    }
                }
            } else {
                in_header = false;
            }
            self.lines.push(parsed_line);
        }

        let mut lines = self.lines.clone();
        for line in lines.iter_mut() {
            match line {
                ParsedLine::Comment => {}
                ParsedLine::Header(_) => {}
                ParsedLine::Label(lab) => {
                    if let Some(_old) = self.linked_refs.insert(lab.to_owned(), self.pc) {
                        return Err(Error::msg(format!("Found duplicate label: {}", lab.name)));
                    }
                }
                ParsedLine::LabelOffsetConst(name, ..) => {
                    if let Some(_old) = self.linked_refs.insert(name.to_owned(), self.pc) {
                        return Err(Error::msg(format!(
                            "Found duplicate const label offset: {}",
                            name.name
                        )));
                    }
                }
                ParsedLine::Data(dat) => {
                    let align_adjust = dat.align - self.pc as usize % dat.align;
                    self.push_program_bytes(&vec![0u8; align_adjust]);
                    if let Some(_old) = self.linked_refs.insert(dat.label.to_owned(), self.pc) {
                        return Err(Error::msg(format!(
                            "Found duplicate data label: {}",
                            dat.label.name
                        )));
                    }
                    self.push_program_bytes(&dat.data);
                }
                ParsedLine::Instr(ins) => {
                    ins.assemble(self)?;
                }
            }
        }

        // parse included files
        for module_path in self.included_modules.clone().iter() {
            let mut file = File::open(module_path)
                .context(format!("Error opening included file `{module_path}`"))?;
            let mut buf = String::new();
            file.read_to_string(&mut buf)
                .context(format!("Error reading included file `{module_path}`"))?;
            let mut ctx = AssemblyContext::new(buf);
            let output = ctx
                .assemble_impl(
                    Some(self.pc),
                    false,
                    false,
                    &self
                        .included_modules
                        .union(existing_includes)
                        .cloned()
                        .collect::<FxHashSet<_>>(),
                    &self.linked_refs,
                )
                .context(format!(
                    "Error while assembling included file `{module_path}`"
                ))?;
            self.push_program_bytes(&output);
            for (ref lab, adr) in ctx.linked_refs {
                if self.linked_refs.get(lab).is_none() {
                    self.linked_refs.insert(lab.to_owned(), adr);
                    // return Err(Error::msg(format!(
                    //     "Found duplicate label `{}` in included file `{module_path}`",
                    //     lab.name
                    // )));
                }
            }
            for ((off, ref lab), adr) in ctx.linked_offset_refs {
                if self
                    .linked_offset_refs
                    .insert((off, lab.to_owned()), adr)
                    .is_some()
                {
                    return Err(Error::msg(format!(
                        "Found duplicate label offset `{}` in included file `{module_path}`",
                        lab.name
                    )));
                }
            }

            for (lab, adr) in ctx.unlinked_refs {
                self.unlinked_refs.insert(lab, adr);
            }
            for (off, adr) in ctx.unlinked_offset_refs {
                self.unlinked_offset_refs.insert(off, adr);
            }
        }

        // one more linking phase, for the forward decls
        for line in lines.iter_mut() {
            if let ParsedLine::Instr(ins) = line {
                if let Some(Token::LabelOffset(off, ref lab)) = ins.arg0 {
                    if let Some(link_location) = self.linked_refs.get(lab) {
                        let ref_locs =
                            self.unlinked_offset_refs.get_mut(&(off, lab.to_owned())).unwrap_or_else(|| {
                                panic!(
                                    "Label offset ({}+{}) needed linking, but wasn't in the unlinked refs",
                                    off,
                                    lab
                                )
                            });
                        for ref_loc in ref_locs.drain() {
                            let ref_loc = ref_loc as usize;
                            self.output[ref_loc - self.entry_point as usize
                                ..ref_loc - self.entry_point as usize + 8]
                                .copy_from_slice(
                                    &((*link_location as i64 + off) as u64).to_bytes(),
                                );
                        }
                    } else if check_link {
                        return Err(Error::msg(format!(
                            "Undefined reference to label {}",
                            lab.name
                        )));
                    }
                } else if let Some(Token::Label(ref mut lab)) = ins.arg0 {
                    if lab.linkage == Linkage::NeedsLinking {
                        if let Some(link_location) = self.linked_refs.get(lab) {
                            let ref_locs = self.unlinked_refs.get_mut(lab).unwrap_or_else(|| {
                                panic!(
                                    "Label {} needed linking, but wasn't in the unlinked refs",
                                    lab
                                )
                            });
                            lab.linkage = Linkage::Linked(*link_location);
                            for ref_loc in ref_locs.drain() {
                                let ref_loc = ref_loc as usize;
                                self.output[ref_loc - self.entry_point as usize
                                    ..ref_loc - self.entry_point as usize + 8]
                                    .copy_from_slice(&link_location.to_bytes());
                            }
                            self.unlinked_refs.remove(lab);
                        } else if check_link {
                            return Err(Error::msg(format!(
                                "Undefined reference to label {}",
                                lab
                            )));
                        }
                    }
                }

                if let Some(Token::LabelOffset(off, ref lab)) = ins.arg1 {
                    if let Some(link_location) = self.linked_refs.get(lab) {
                        let ref_locs =
                            self.unlinked_offset_refs.get_mut(&(off, lab.to_owned())).unwrap_or_else(|| {
                                panic!(
                                    "Label offset ({}+{}) needed linking, but wasn't in the unlinked refs",
                                    off,
                                    lab
                                )
                            });
                        for ref_loc in ref_locs.drain() {
                            let ref_loc = ref_loc as usize;
                            self.output[ref_loc - self.entry_point as usize
                                ..ref_loc - self.entry_point as usize + 8]
                                .copy_from_slice(
                                    &((*link_location as i64 + off) as u64).to_bytes(),
                                );
                        }
                        // self.unlinked_offset_refs.remove(&(off, lab.to_owned()));
                    } else if check_link {
                        return Err(Error::msg(format!(
                            "Undefined reference to label {}",
                            lab.name
                        )));
                    }
                } else if let Some(Token::Label(ref mut lab)) = ins.arg1 {
                    if lab.linkage == Linkage::NeedsLinking {
                        if let Some(link_location) = self.linked_refs.get(lab) {
                            let ref_locs = self.unlinked_refs.get_mut(lab).unwrap_or_else(|| {
                                panic!(
                                    "Label {} needed linking, but wasn't in the unlinked refs",
                                    lab.name
                                )
                            });
                            lab.linkage = Linkage::Linked(*link_location);
                            for ref_loc in ref_locs.drain() {
                                let ref_loc = ref_loc as usize;
                                self.output[ref_loc - self.entry_point as usize
                                    ..ref_loc - self.entry_point as usize + 8]
                                    .copy_from_slice(&link_location.to_bytes());
                            }
                            self.unlinked_refs.remove(lab);
                        } else if check_link {
                            return Err(Error::msg(format!(
                                "Undefined reference to label {}",
                                lab.name
                            )));
                        }
                    }
                }
            }
        }

        if check_link
            && !self.unlinked_refs.is_empty()
            && !self.unlinked_refs.values().all(|refs| refs.is_empty())
        {
            for (lab, refs) in self.unlinked_refs.iter() {
                if !refs.is_empty() {
                    log::error!(
                        "Undefined reference to label {} after stage 2 of linking",
                        lab.name
                    );
                }
            }
            return Err(Error::msg(
                "Undefined references to labels after stage 2 of linking",
            ));
        }

        if include_header {
            let mut final_out = Vec::new();
            final_out.extend_from_slice(tags::HEADER_MAGIC);
            final_out.extend_from_slice(tags::HEADER_ENTRY_POINT);
            final_out.extend_from_slice(&self.entry_point.to_bytes());
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_START);
            for (label, addr) in self.linked_refs.iter() {
                final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_ADDR);
                final_out.extend_from_slice(&addr.to_bytes());
                final_out.extend_from_slice(label.name.as_bytes());
                final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_END);
            }
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_END);
            final_out.extend_from_slice(tags::HEADER_END);

            final_out.extend_from_slice(&self.output);
            self.output = final_out;
            // println!("Assembled program is {} bytes long.", self.output.len());

            Ok(self.output.clone())
        } else {
            Ok(self.output.clone())
        }
    }
}
