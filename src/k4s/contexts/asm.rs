use anyhow::{Context, Error, Result};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::k4s::{
    parsers::{
        asm::{data, header, header_entry, lab_offset_const, label, literal, opcode, size, token},
        machine::tags,
    },
    Data, Instr, InstrSize, Label, Linkage, Opcode, Primitive, Token,
};

#[derive(Debug, Clone)]
pub enum Header {
    Entry(Label, u64),
    Include(String),
}

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Comment(String),
    Header(Header),
    Label(Label),
    Data(Data),
    LabelOffsetConst(Label, i64, Label),
    Instr(Instr),
}

impl ParsedLine {
    pub fn is_term(&self) -> bool {
        matches!(
            self,
            ParsedLine::Instr(
                Instr {
                    opcode: Opcode::Ret,
                    ..
                } | Instr {
                    opcode: Opcode::Hlt,
                    ..
                }
            ),
        )
    }

    pub fn display(&self, symbols: &FxHashMap<u64, String>) -> String {
        match self {
            Self::Comment(comment) => comment.to_owned(),
            Self::Header(head) => match head {
                Header::Include(path) => format!("!include \"{}\"", path),
                Header::Entry(lab, adr) => format!("!ent {} @ {}", lab, adr),
            },
            Self::Label(lab) => format!("{}", lab),
            Self::Data(dat) => format!("{} align{} <data>", dat.label, dat.align),
            Self::LabelOffsetConst(name, off, lab) => format!("@{} ({}+{})", name.name, off, lab),
            Self::Instr(instr) => format!("    {}", instr.display_with_symbols(symbols)),
        }
    }

    pub fn parse(line: &str) -> anyhow::Result<(&str, AssembledLine)> {
        let line = line.trim();
        if line.starts_with(';') {
            return Ok((
                line,
                AssembledLine::new_unassembled(ParsedLine::Comment(line.to_owned())),
            ));
        }
        if let Ok((rest, header)) = header(line) {
            let rest = rest.trim();
            let header = match header {
                "!ent" => {
                    // if rest.starts_with("0x") {
                    //     let rest = rest.strip_prefix("0x").unwrap_or(rest);
                    //     Header::Entry(u64::from_str_radix(rest, 16).unwrap())
                    // } else {
                    //     return Err(Error::msg("Expected hex number after `!ent` tag"));
                    // }
                    if let Ok((_rest, header)) = header_entry(line) {
                        header
                    } else {
                        return Err(Error::msg(
                            "Expected `!ent` tag to be in the form of `!ent %label @ 0xaddr",
                        ));
                    }
                }
                "!include" => {
                    let rest = rest.strip_prefix('\"').unwrap_or(rest);
                    let rest = rest.strip_suffix('\"').unwrap_or(rest);
                    Header::Include(rest.to_owned())
                }
                _ => return Err(Error::msg(format!("Invalid header tag: {header}"))),
            };
            Ok((
                "",
                AssembledLine::new_unassembled(ParsedLine::Header(header)),
            ))
        } else if let Ok((rest, (name, lab_off))) = lab_offset_const(line) {
            if let Token::LabelOffset(off, lab) = lab_off {
                Ok((
                    rest,
                    AssembledLine::new_unassembled(ParsedLine::LabelOffsetConst(name, off, lab)),
                ))
            } else {
                unreachable!()
            }
        } else if let Ok((rest, label)) = label(line) {
            if let Token::Label(label) = label {
                Ok((
                    rest,
                    AssembledLine::new_unassembled(ParsedLine::Label(label)),
                ))
            } else {
                unreachable!()
            }
        } else if let Ok((rest, data)) = data(line) {
            if let Token::Data(data) = data {
                Ok((rest, AssembledLine::new_unassembled(ParsedLine::Data(data))))
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
                    AssembledLine::new_unassembled(ParsedLine::Instr(Instr {
                        opcode,
                        size,
                        arg0,
                        arg1,
                    })),
                ))
            } else {
                Ok((
                    rest,
                    AssembledLine::new_unassembled(ParsedLine::Instr(Instr {
                        opcode,
                        size: InstrSize::Unsized,
                        arg0: None,
                        arg1: None,
                    })),
                ))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssembledLine {
    pub asm: ParsedLine,
    pub mc: Vec<u8>,
}

impl AssembledLine {
    pub fn new_unassembled(asm: ParsedLine) -> Self {
        Self {
            asm,
            mc: Vec::new(),
        }
    }

    pub fn references(&self, label: &str) -> bool {
        if let ParsedLine::Instr(ref instr) = self.asm {
            if let Some(Token::Label(ref arg0_lab)) = instr.arg0 {
                if arg0_lab.name == label {
                    return true;
                }
            }
            if let Some(Token::Label(ref arg0_lab)) = instr.arg1 {
                if arg0_lab.name == label {
                    return true;
                }
            }
        }
        false
    }

    pub fn all_label_refs(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let ParsedLine::Instr(ref instr) = self.asm {
            if let Some(Token::Label(ref arg0_lab)) = instr.arg0 {
                out.push(arg0_lab.name.to_owned());
            }
            if let Some(Token::Label(ref arg1_lab)) = instr.arg1 {
                out.push(arg1_lab.name.to_owned());
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct AssembledBlock {
    pub name: String,
    pub refs: usize,
    pub loc: u64,
    pub lines: Vec<AssembledLine>,
}

impl AssembledBlock {
    pub fn references(&self, other_name: &str) -> bool {
        self.lines.iter().any(|line| line.references(other_name))
    }

    pub fn all_label_refs(&self) -> Vec<String> {
        let mut out = Vec::new();
        for line in self.lines.iter() {
            out.extend_from_slice(&line.all_label_refs());
        }
        out
    }
}

pub struct AssemblyContext {
    pub blocks: FxHashMap<String, AssembledBlock>,
    pub linked_refs: FxHashMap<String, u64>, // (label, addr found at)
    pub unlinked_refs: FxHashMap<String, FxHashSet<u64>>, // (label, addrs found at)
    pub linked_offset_refs: FxHashMap<(i64, Label), u64>, // ((offset, label), addr found at)
    pub unlinked_offset_refs: FxHashMap<(i64, Label), FxHashSet<u64>>, // ((offset, label), addrs found at)

    pub entry_label: Option<Label>,
    pub used_labels: FxHashSet<usize>,

    pub included_modules: FxHashSet<String>,
    pub input: String,
    pub pc: u64,
    pub entry_point: u64,

    pub symbols: FxHashMap<u64, String>,
    pub kept_blocks: Vec<AssembledBlock>,
    pub output: Vec<u8>,
}

impl AssemblyContext {
    pub fn new(input: String) -> Self {
        Self {
            blocks: FxHashMap::default(),
            linked_refs: FxHashMap::default(),
            unlinked_refs: FxHashMap::default(),
            linked_offset_refs: FxHashMap::default(),
            unlinked_offset_refs: FxHashMap::default(),
            included_modules: FxHashSet::default(),
            entry_label: None,
            used_labels: FxHashSet::default(),
            input,
            pc: 0,
            entry_point: 0,
            symbols: FxHashMap::default(),
            // kept_lines: Vec::new(),
            kept_blocks: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn push_program_bytes(&mut self, bytes: &[u8]) {
        self.output.extend_from_slice(bytes);
        self.pc += bytes.len() as u64;
    }

    pub fn assemble(&mut self) -> Result<Vec<u8>> {
        self.assemble_impl(None, true, &FxHashSet::default(), &FxHashMap::default())
        // todo: re-enable link checking
    }

    fn assemble_impl(
        &mut self,
        entry_point: Option<u64>,
        include_header: bool,
        existing_includes: &FxHashSet<String>,
        existing_linked_refs: &FxHashMap<String, u64>,
    ) -> Result<Vec<u8>> {
        if let Some(entry_point) = entry_point {
            self.entry_point = entry_point;
            self.pc = entry_point;
        }

        self.linked_refs = existing_linked_refs.clone();

        let mut in_header = true;
        let mut in_function = false;
        let mut current_function = None;
        let mut data_lines = Vec::default();
        for (line_no, line) in self.input.clone().lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let line_no = line_no + 1;
            let (junk, mut parsed_line) = ParsedLine::parse(line)
                .context(format!("Error while parsing line {}:\n{}", line_no, line))?;
            if let ParsedLine::Data(ref mut data) = parsed_line.asm {
                // parse the junk as data
                let string = junk.trim();
                if let Ok((_junk, lit)) = literal(InstrSize::I64, string) {
                    let lit = lit.as_integer::<u64>().unwrap();
                    data.data = lit.to_bytes().to_vec();
                } else if let Some(amount) = string.strip_prefix("resb") {
                    let amount = amount.trim();
                    let amount = if let Some(amount) = amount.strip_prefix("0x") {
                        u64::from_str_radix(amount, 16).unwrap()
                    } else {
                        amount.parse::<u64>().unwrap()
                    };
                    data.data.extend_from_slice(&vec![0u8; amount as usize]);
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
                data_lines.push(parsed_line.to_owned());
                // self.push_program_bytes(&data.data);
            } else if !junk.is_empty() && !junk.trim().starts_with(';') {
                log::warn!("Ignoring junk after line {line_no}: {junk}");
            }
            if let ParsedLine::Header(ref header) = parsed_line.asm {
                if !in_header {
                    return Err(Error::msg(format!(
                        "Found header line outside of header on line {line_no}: {line}"
                    )));
                }

                match header {
                    Header::Entry(lab, pc) => {
                        if entry_point.is_some() {
                            return Err(Error::msg(format!(
                                "Found entry point header line in non-main module on line {line_no}: {line}"
                            )));
                        }
                        self.entry_label = Some(lab.to_owned());
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
                if let ParsedLine::Label(ref lab) = parsed_line.asm {
                    if !in_function {
                        in_function = true;
                        current_function = Some(AssembledBlock {
                            name: lab.name.to_owned(),
                            refs: 0,
                            loc: 0,
                            lines: Vec::new(),
                        });
                    }
                }
                if in_function {
                    current_function
                        .as_mut()
                        .unwrap()
                        .lines
                        .push(parsed_line.to_owned());
                }
                if let ParsedLine::Instr(Instr {
                    opcode: Opcode::Hlt | Opcode::Ret,
                    ..
                }) = parsed_line.asm
                {
                    if in_function {
                        in_function = false;
                        let func = current_function.as_ref().unwrap().to_owned();
                        self.blocks.insert(func.name.to_owned(), func);
                        current_function = None;
                    }
                }
            }
        }

        for (_name, block) in self.blocks.clone().into_iter() {
            for reference in block.all_label_refs() {
                if let Some(referenced_block) = self.blocks.get_mut(&reference) {
                    referenced_block.refs += 1;
                }
            }
        }

        let mut kept_blocks = self
            .blocks
            .iter()
            .filter(|(name, block)| {
                block.refs > 0 || name == &&self.entry_label.as_ref().unwrap().name
            })
            .map(|(_, block)| block.to_owned())
            .collect::<Vec<_>>();
        kept_blocks.sort_by_key(|block| block.name != self.entry_label.as_ref().unwrap().name);

        let thrown_away_count = self.blocks.len() - kept_blocks.len();
        log::debug!("Threw away {} unused blocks!", thrown_away_count);

        self.pc = self.entry_point;

        for block in kept_blocks.iter_mut() {
            if block.name == self.entry_label.as_ref().unwrap().name {
                log::trace!(
                    "Assembling block {} (entry point, {} lines)",
                    block.name,
                    block.lines.len()
                );
            } else {
                log::trace!(
                    "Assembling block {} ({} refs, {} lines)",
                    block.name,
                    block.refs,
                    block.lines.len()
                );
            }
            block.loc = self.pc;
            for line in block.lines.iter_mut() {
                if let ParsedLine::Instr(ref mut instr) = line.asm {
                    let (size, refs) = instr.assemble(self.pc, &mut line.mc)?;
                    for (refr, loc) in refs {
                        self.unlinked_refs
                            .entry(refr.name)
                            .or_insert(FxHashSet::default())
                            .insert(loc);
                    }
                    self.pc += size as u64;
                } else if let ParsedLine::Label(ref mut lab) = line.asm {
                    if !self.linked_refs.contains_key(&lab.name) {
                        lab.linkage = Linkage::Linked(self.pc);
                        self.linked_refs.insert(lab.name.to_owned(), self.pc);
                    }
                }
            }
        }

        self.pc = self.entry_point;
        for block in kept_blocks.iter_mut() {
            for line in block.lines.iter_mut() {
                self.push_program_bytes(&line.mc);
            }
        }
        for data_line in data_lines {
            if let ParsedLine::Data(data) = data_line.asm {
                if !self.linked_refs.contains_key(&data.label.name) {
                    if data.align > 0 {
                        self.push_program_bytes(&vec![
                            0;
                            data.align - self.pc as usize % data.align
                        ]);
                    }
                    self.linked_refs
                        .insert(data.label.name.to_owned(), self.pc + 1);
                    self.push_program_bytes(&data.data);
                }
            }
        }

        log::debug!("Found {} linked references.", self.linked_refs.len());

        let mut undefined_count: usize = 0;

        for (lab, refs) in self.unlinked_refs.iter_mut() {
            if let Some(loc) = self.linked_refs.get(lab) {
                for refr in refs.drain() {
                    self.output[refr as usize - self.entry_point as usize
                        ..refr as usize - self.entry_point as usize + 8]
                        .copy_from_slice(&loc.to_bytes());
                }
            } else {
                undefined_count += 1;
            }
        }

        log::debug!("{} unlinked references remain.", undefined_count);

        self.kept_blocks = kept_blocks;

        let actual_entry_point = self
            .kept_blocks
            .iter()
            .find_map(|block| {
                if self.entry_label.as_ref().unwrap().name == block.name {
                    Some(block.loc)
                } else {
                    None
                }
            })
            .unwrap();

        self.entry_point = actual_entry_point;

        if include_header {
            let mut final_out = Vec::new();
            final_out.extend_from_slice(tags::HEADER_MAGIC);
            final_out.extend_from_slice(tags::HEADER_ENTRY_POINT);
            final_out.extend_from_slice(&self.entry_point.to_bytes());
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_START);
            for (label, addr) in self.linked_refs.iter() {
                final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_ADDR);
                final_out.extend_from_slice(&addr.to_bytes());
                final_out.extend_from_slice(label.as_bytes());
                final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_END);

                self.symbols.insert(*addr, label.to_owned());
            }
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_END);
            final_out.extend_from_slice(tags::HEADER_END);

            final_out.extend_from_slice(&self.output);
            self.output = final_out;

            Ok(self.output.clone())
        } else {
            Ok(self.output.clone())
        }
    }
}
