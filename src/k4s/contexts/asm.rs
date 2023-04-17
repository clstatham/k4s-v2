use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{Context, Error, Result};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::k4s::{
    parsers::{
        asm::{
            data, header, header_entry, header_region, lab_offset_const, label, opcode, size,
            token, UnlinkedRef, UnlinkedRefType,
        },
        machine::tags,
    },
    Data, Instr, InstrSize, Label, Opcode, Primitive, Token,
};

#[derive(Debug, Clone)]
pub enum Header {
    Entry(Label, u64),
    Include(String),
    Region(u64, u64, Vec<Label>),
}

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Comment(String),
    Header(Header),
    Label(Label),
    Data(Data),
    DataOffsetConst(Label, i64, Label),
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

    pub fn display(&self, symbols: &BTreeMap<u64, String>) -> String {
        match self {
            Self::Comment(comment) => comment.to_owned(),
            Self::Header(head) => match head {
                Header::Include(path) => format!("!include \"{}\"", path),
                Header::Entry(lab, adr) => format!("!ent {} @ {}", lab, adr),
                Header::Region(virt, load, _labels) => {
                    format!("!region {} > {} : <labels>", virt, load)
                }
            },
            Self::Label(lab) => format!("{}", lab),
            Self::Data(dat) => format!("{} align{} <data>", dat.label, dat.align),
            Self::DataOffsetConst(name, off, lab) => format!("@{} ({}+{})", name.name(), off, lab),
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
                "!region" => {
                    if let Ok((_rest, header)) = header_region(line) {
                        header
                    } else {
                        return Err(Error::msg(
                            "Expected `!region` tag to be in the form of `!region 0xvirtaddr > 0xloadaddr : label0 label1 (...)`",
                        ));
                    }
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
                    AssembledLine::new_unassembled(ParsedLine::DataOffsetConst(name, off, lab)),
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

    pub fn all_label_refs(&mut self) -> Vec<&mut Label> {
        let mut out = Vec::new();
        if let ParsedLine::Instr(ref mut instr) = self.asm {
            if let Some(Token::Label(ref mut arg0_lab)) = instr.arg0 {
                out.push(arg0_lab);
            }
            if let Some(Token::Label(ref mut arg1_lab)) = instr.arg1 {
                out.push(arg1_lab);
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct AssembledBlock {
    pub label: Label,
    pub refs: usize,
    pub lines: Vec<AssembledLine>,
}

impl AssembledBlock {
    pub fn all_label_refs(&mut self) -> Vec<&mut Label> {
        let mut out = Vec::new();
        for line in self.lines.iter_mut() {
            out.extend(line.all_label_refs());
        }
        out
    }
}

#[derive(Debug)]
pub struct MemoryRegion {
    pub id: usize,
    pub data: Vec<u8>,
    pub program: Vec<u8>,
    pub rel_pc: u64,
    pub virt: u64,
    pub load: u64,
}

#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum RefType {
    Label(String),
    Data(String),
}

impl RefType {
    pub fn name(&self) -> String {
        match self {
            Self::Label(n) => n.to_owned(),
            Self::Data(n) => n.to_owned(),
        }
    }
}

pub struct AssemblyContext {
    pub blocks: FxHashMap<String, AssembledBlock>,
    pub linked_refs: FxHashMap<RefType, u64>, // (label, addr found at)
    pub unlinked_refs: FxHashMap<RefType, Vec<UnlinkedRef>>, // (label, addrs found at)

    pub region_mappings: FxHashMap<String, usize>,
    pub regions: FxHashMap<usize, MemoryRegion>,
    pub next_region_id: AtomicUsize,

    pub entry_label: Option<Label>,
    pub used_labels: FxHashSet<usize>,

    pub included_modules: FxHashSet<String>,
    pub input: String,
    pub pc: u64,
    pub entry_point: u64,

    pub symbols: BTreeMap<u64, String>,
    pub kept_blocks: Vec<AssembledBlock>,
    pub output: Vec<u8>,
}

impl AssemblyContext {
    pub fn new(input: String) -> Self {
        Self {
            blocks: FxHashMap::default(),
            linked_refs: FxHashMap::default(),
            unlinked_refs: FxHashMap::default(),
            included_modules: FxHashSet::default(),
            region_mappings: FxHashMap::default(),
            regions: FxHashMap::default(),
            next_region_id: AtomicUsize::new(0),
            entry_label: None,
            used_labels: FxHashSet::default(),
            input,
            pc: 0,
            entry_point: 0,
            symbols: BTreeMap::default(),
            kept_blocks: Vec::new(),
            output: Vec::new(),
        }
    }

    pub fn push_program_bytes(&mut self, bytes: &[u8], region_id: usize) {
        self.regions
            .get_mut(&region_id)
            .unwrap()
            .program
            .extend_from_slice(bytes);
    }

    pub fn push_data_bytes(&mut self, bytes: &[u8], region_id: usize) {
        self.regions
            .get_mut(&region_id)
            .unwrap()
            .data
            .extend_from_slice(bytes);
    }

    // todo: includes
    pub fn assemble(&mut self, include_symbols: bool) -> Result<Vec<u8>> {
        self.assemble_impl(
            None,
            true,
            include_symbols,
            &FxHashSet::default(),
            &FxHashMap::default(),
        )
    }

    fn assemble_impl(
        &mut self,
        entry_point: Option<u64>,
        include_header: bool,
        include_symbols: bool,
        existing_includes: &FxHashSet<String>,
        existing_linked_refs: &FxHashMap<RefType, u64>,
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
            let (junk, parsed_line) = ParsedLine::parse(line)
                .context(format!("Error while parsing line {}:\n{}", line_no, line))?;
            if let ParsedLine::Data(_) = parsed_line.asm {
                data_lines.push(parsed_line.to_owned());
            } else if let ParsedLine::DataOffsetConst(..) = parsed_line.asm {
                data_lines.push(parsed_line.to_owned());
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
                        // let region = MemoryRegion {
                        //     id: self.next_region_id.fetch_add(1, Ordering::SeqCst),
                        //     data: Vec::new(),
                        //     rel_pc: 0,
                        //     virt: *pc,
                        //     load: *pc,
                        // };
                        // assert_eq!(region.id, 0);
                        // self.regions.insert(region.id, region);
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
                    Header::Region(virt, load, labels) => {
                        let region = MemoryRegion {
                            id: self.next_region_id.fetch_add(1, Ordering::SeqCst),
                            program: Vec::new(),
                            data: Vec::new(),
                            rel_pc: 0,
                            virt: *virt,
                            load: *load,
                        };
                        for lab in labels {
                            self.region_mappings.insert(lab.name(), region.id);
                        }
                        self.regions.insert(region.id, region);
                    }
                }
            } else {
                in_header = false;
                if let ParsedLine::Label(ref lab) = parsed_line.asm {
                    if !in_function {
                        in_function = true;
                        current_function = Some(AssembledBlock {
                            label: lab.clone(),
                            refs: 0,
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
                        self.blocks.insert(func.label.name(), func);
                        current_function = None;
                    }
                }
            }
        }

        for (_name, block) in self.blocks.clone().iter_mut() {
            for reference in block.all_label_refs() {
                if let Some(referenced_block) = self.blocks.get_mut(&reference.name()) {
                    referenced_block.refs += 1;
                }
            }
        }

        // todo: reimplement dead code pruning
        let mut kept_blocks: Vec<AssembledBlock> = self
            .blocks
            .values()
            .map(|block| block.to_owned())
            .collect::<Vec<_>>();
        kept_blocks
            .sort_by_key(|block| block.label.name() != self.entry_label.as_ref().unwrap().name());

        let thrown_away_count = self.blocks.len() - kept_blocks.len();
        log::debug!("Threw away {} unused blocks!", thrown_away_count);

        // assemble each block within the context of its memory region (addresses are relative to the region's start)
        for block in kept_blocks.iter_mut() {
            if block.label.name() == self.entry_label.as_ref().unwrap().name() {
                log::trace!(
                    "Assembling block {} (entry point, {} lines)",
                    block.label,
                    block.lines.len()
                );
            } else {
                log::trace!(
                    "Assembling block {} ({} refs, {} lines)",
                    block.label,
                    block.refs,
                    block.lines.len()
                );
            }
            // default the region to the entry point's
            {
                let region = self
                    .regions
                    .get(self.region_mappings.get(&block.label.name()).unwrap_or(&0))
                    .unwrap();
                block.label.region_id = Some(region.id);
                self.region_mappings.insert(block.label.name(), region.id);
            }

            // for every line of assembly...
            for line in block.lines.iter_mut() {
                let region = self
                    .regions
                    .get_mut(self.region_mappings.get(&block.label.name()).unwrap_or(&0))
                    .unwrap();

                // if it's an instruction, make sure we link any addrs in the arguments later
                if let ParsedLine::Instr(ref mut instr) = line.asm {
                    let (size, mut refs) =
                        instr.assemble(region.rel_pc, region.id, &mut line.mc)?;

                    // put addr arguments in the unlinked_refs to the label for that addr
                    for refr in refs.iter_mut() {
                        refr.label.region_id =
                            self.region_mappings.get(&refr.label.name()).copied();
                        self.unlinked_refs
                            .entry(refr.ref_type())
                            .or_insert(Vec::default())
                            .push(refr.to_owned());
                    }

                    region.rel_pc += size as u64;
                } else if let ParsedLine::Label(ref mut lab) = line.asm {
                    // if it's a whole label, mark down where it ended up, offset by the region's virtual start
                    lab.region_id = Some(region.id);
                    self.region_mappings.insert(lab.name(), region.id);
                    self.linked_refs
                        .entry(RefType::Label(lab.name()))
                        .or_insert(region.rel_pc);
                }
            }
        }

        for data_line in data_lines {
            if let ParsedLine::Data(data) = data_line.asm {
                if !self
                    .linked_refs
                    .contains_key(&RefType::Data(data.label.name()))
                {
                    let region_id = *self.region_mappings.entry(data.label.name()).or_insert(0);
                    let region = self.regions.get(&region_id).unwrap();

                    if data.align > 0 {
                        let pad = (data.align - (region.data.len() % data.align)) % data.align;
                        self.push_data_bytes(&vec![0; pad], region_id);
                    }

                    let region = self.regions.get(&region_id).unwrap();
                    self.linked_refs.insert(
                        RefType::Data(data.label.name().to_owned()),
                        region.data.len() as u64,
                    );

                    self.push_data_bytes(&data.data, region_id);
                }
            } else if let ParsedLine::DataOffsetConst(name, off, lab) = data_line.asm {
                let region_id = *self.region_mappings.entry(name.name()).or_insert(0);
                let region = self.regions.get(&region_id).unwrap();

                self.unlinked_refs
                    .entry(RefType::Data(lab.name()))
                    .or_insert(Vec::default())
                    .push(UnlinkedRef {
                        ty: UnlinkedRefType::DataOffset(off),
                        label: lab.to_owned(),
                        region_id,
                        loc: region.data.len() as u64,
                    });

                if !self.linked_refs.contains_key(&RefType::Data(name.name())) {
                    self.linked_refs.insert(
                        RefType::Data(name.name().to_owned()),
                        region.data.len() as u64,
                    );
                }

                self.push_data_bytes(&[0; 8], region_id);
            }
        }

        // 2nd pass of resolving label regions
        for (_, refrs) in self.unlinked_refs.iter_mut() {
            for refr in refrs.iter_mut() {
                refr.label.region_id = self.region_mappings.get(&refr.label.name()).copied();
                if refr.label.region_id.is_none() {
                    log::warn!(
                        "2nd pass: Couldn't find or infer region for label {}",
                        refr.label
                    );
                }
            }
        }

        for block in kept_blocks.iter_mut() {
            for refr_label in block.all_label_refs() {
                if refr_label.region_id.is_none() {
                    refr_label.region_id = Some(
                        self.region_mappings
                            .get(&refr_label.name())
                            .copied()
                            .unwrap(),
                    );
                }
            }
        }

        for block in kept_blocks.iter_mut() {
            for line in block.lines.iter_mut() {
                self.push_program_bytes(&line.mc, self.region_mappings[&block.label.name()]);
            }
        }

        log::debug!("Found {} linked references.", self.linked_refs.len());

        let mut undefined_refs = Vec::new();

        for (ref lab, ref mut refs) in self.unlinked_refs.drain() {
            if let Some(loc) = self.linked_refs.get(lab) {
                for refr in refs.drain(..) {
                    let (adjusted_loc, program_len) = {
                        let region = self.regions.get(&refr.label.region_id.unwrap()).unwrap();
                        (*loc + region.virt, region.program.len() as u64)
                    };

                    let refr_region = self.regions.get_mut(&refr.region_id).unwrap();

                    match refr.ty {
                        UnlinkedRefType::Label => refr_region.program
                            [refr.loc as usize..refr.loc as usize + 8]
                            .copy_from_slice(&(adjusted_loc).to_bytes()),
                        UnlinkedRefType::DataOffset(off) => {
                            let adjusted_loc = adjusted_loc + program_len;
                            refr_region.data[refr.loc as usize..refr.loc as usize + 8]
                                .copy_from_slice(&((adjusted_loc as i64 + off) as u64).to_bytes());
                        }
                        UnlinkedRefType::Data => {
                            let adjusted_loc = adjusted_loc + program_len;
                            refr_region.program[refr.loc as usize..refr.loc as usize + 8]
                                .copy_from_slice(&adjusted_loc.to_bytes());
                        }
                    }
                }
            } else {
                undefined_refs.push(lab.to_owned());
            }
        }

        log::debug!("{} undefined references remain.", undefined_refs.len());
        for refr in undefined_refs {
            log::debug!("Undefined reference to {:?}", refr);
        }

        let binary_size = self
            .regions
            .values()
            .map(|reg| reg.load as usize + reg.program.len() + reg.data.len())
            .max()
            .unwrap();
        let mut ordered_regions = self.regions.iter().collect::<Vec<_>>();
        ordered_regions.sort_by_key(|(_, region)| region.load);

        self.output = vec![0u8; binary_size];
        for (_, region) in ordered_regions {
            self.output[region.load as usize..region.load as usize + region.program.len()]
                .copy_from_slice(&region.program);
            self.output[region.load as usize + region.program.len()
                ..region.load as usize + region.program.len() + region.data.len()]
                .copy_from_slice(&region.data);
        }

        self.kept_blocks = kept_blocks;

        if include_header {
            let mut final_out = Vec::new();
            final_out.extend_from_slice(tags::HEADER_MAGIC);
            final_out.extend_from_slice(tags::HEADER_ENTRY_POINT);
            final_out.extend_from_slice(&self.entry_point.to_bytes());
            final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_START);
            if include_symbols {
                let mut syms = self
                    .linked_refs
                    .iter()
                    .map(|(s, adr)| {
                        let region = self
                            .regions
                            .get(self.region_mappings.get(&s.name()).unwrap())
                            .unwrap();
                        (s, adr + region.virt)
                    })
                    .collect::<Vec<_>>();
                syms.sort_by_key(|(_, sym_start)| *sym_start);
                for (label, addr) in syms {
                    final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_ADDR);
                    final_out.extend_from_slice(&addr.to_bytes());
                    final_out.extend_from_slice(label.name().as_bytes());
                    final_out.extend_from_slice(tags::HEADER_DEBUG_SYMBOLS_ENTRY_END);

                    self.symbols.insert(addr, label.name());
                }
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
