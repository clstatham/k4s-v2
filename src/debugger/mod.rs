use std::{fs::File, io::Read, path::PathBuf};

use crate::{k4s::contexts::machine::MachineContext, FOUR_GIGS};
use eframe::{egui, epaint::Vec2, NativeOptions};

pub fn debugger_main(bin_paths: Vec<PathBuf>) {
    eframe::run_native(
        "k4s debugger",
        NativeOptions {
            initial_window_size: Some(Vec2::new(1600.0, 900.0)),
            ..Default::default()
        },
        Box::new(move |_cc| Box::new(DebuggerApp::new(&bin_paths))),
    )
    .unwrap();
}

pub struct DebuggerApp {
    emu: MachineContext,
}

impl DebuggerApp {
    pub fn new(bin_paths: &[PathBuf]) -> Self {
        let mut programs = vec![];
        for bin_path in bin_paths {
            let mut file = File::open(bin_path).unwrap();
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).unwrap();
            programs.push(buf);
        }
        let emu = MachineContext::new(programs, FOUR_GIGS).unwrap();
        Self { emu }
    }
}

impl eframe::App for DebuggerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("registers").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Continue").clicked() {
                            if let Err(e) = self.emu.run_until_hlt() {
                                log::error!("{:?}", e);
                            }
                            self.emu.update_dbg_info().unwrap();
                        }
                        if ui.button("Step").clicked() {
                            if let Err(e) = self.emu.step(true) {
                                log::error!("{:?}", e);
                            }
                        }
                    });
                });
                let regs_text = self.emu.regs.to_string();
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut regs_text.as_str()).code_editor(),
                );
            });
        });
        egui::TopBottomPanel::bottom("errors").show(ctx, |ui| {
            let errors = self.emu.error.as_ref().cloned().unwrap_or(String::new());
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::singleline(&mut errors.as_str()),
            );
        });

        egui::SidePanel::left("output").show(ctx, |ui| {
            let output = self.emu.output_history.to_owned();
            ui.add_sized(
                ui.available_size(),
                egui::TextEdit::multiline(&mut output.as_str()).code_editor(),
            );
        });
        egui::SidePanel::right("stack").show(ctx, |ui| {
            ui.vertical(|ui| {
                let stack_text = {
                    self.emu
                        .stack_frame
                        .iter()
                        .copied()
                        .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
                        .collect::<Vec<String>>()
                        .join("\n")
                };
                ui.group(|ui| {
                    egui::containers::ScrollArea::both()
                        .id_source("stack_text")
                        .show(ui, |ui| {
                            ui.add_sized(
                                ui.available_size() * Vec2::new(1.0, 0.5),
                                egui::TextEdit::multiline(&mut stack_text.as_str()).code_editor(),
                            );
                        });
                });

                let call_stack_text = self.emu.call_stack.join("\n");
                ui.group(|ui| {
                    egui::containers::ScrollArea::both()
                        .id_source("call_stack_text")
                        .show(ui, |ui| {
                            ui.add_sized(
                                ui.available_size(),
                                egui::TextEdit::multiline(&mut call_stack_text.as_str())
                                    .code_editor(),
                            );
                        });
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let instr_text = self
                .emu
                .instr_history
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            egui::containers::ScrollArea::both().show(ui, |ui| {
                ui.add_sized(
                    ui.available_size(),
                    egui::TextEdit::multiline(&mut instr_text.as_str()).code_editor(),
                );
            });
        });
    }
}

// #[allow(non_snake_case)]
// fn App(cx: Scope<DebuggerProps>) -> Element {
//     let mut programs = vec![];
//     for bin_path in &cx.props.bin_paths {
//         let mut file = File::open(bin_path).unwrap();
//         let mut buf = Vec::new();
//         file.read_to_end(&mut buf).unwrap();
//         programs.push(buf);
//     }
//     let emu = use_ref(cx, || MachineContext::new(programs, FOUR_GIGS).unwrap());
//     let style = rsx! {
//         style { include_str!("assets/style.css") }
//     };
//     let regs_text = { emu.read().regs.to_string() };
//     let registers =
//         rsx! { textarea { class: "debug_textarea regs", readonly: true, "{regs_text}" } };
//     let instructions_text = {
//         emu.read()
//             .instr_history
//             .iter()
//             .cloned()
//             .collect::<Vec<_>>()
//             .join("\n")
//     };
//     let instructions =
//         rsx! {textarea { class: "debug_textarea instrs", readonly: true, "{instructions_text}" }};

//     let step_btn = rsx! {
//         div { class: "controls",
//             button {
//                 onclick: move |_| {
//                     if let Err(e) = emu.write().step(true) {
//                         log::debug!("{:?}", e);
//                     }
//                 },
//                 "Step"
//             }
//             button {
//                 onclick: move |_| {
//                     let mut emu = emu.write();
//                     if let Err(e) = emu.run_until_hlt() {
//                         log::debug!("{:?}", e);
//                     }
//                     emu.update_dbg_info().unwrap();
//                 },
//                 "Continue"
//             }
//         }
//     };

//     let stack_text = {
//         emu.read()
//             .stack_frame
//             .iter()
//             .copied()
//             .map(|(adr, val)| format!("bp - {:>5}: {:016x}", adr, val))
//             .collect::<Vec<String>>()
//             .join("\n")
//     };

//     let stack = rsx! { textarea { class: "debug_textarea stack", readonly: true, "{stack_text}" } };

//     let call_stack_text = { emu.read().call_stack.join("\n") };
//     let calls =
//         rsx! { textarea { class: "debug_textarea calls", readonly: true, "{call_stack_text}" } };

//     let errors_text = {
//         let lock = emu.read();
//         lock.error.as_ref().cloned().unwrap_or(String::new())
//     };
//     let errors =
//         rsx! { textarea { class: "debug_textarea errors", readonly: true, "{errors_text}" } };

//     let output_text = {
//         let lock = emu.read();
//         lock.output_history.to_owned()
//     };
//     let output =
//         rsx! { textarea { class: "debug_textarea output", readonly: true, "{output_text}" } };

//     cx.render(rsx! {
//         style,
//         div { class: "grid-container", step_btn, registers, instructions, stack, calls, errors, output }
//     })
// }
