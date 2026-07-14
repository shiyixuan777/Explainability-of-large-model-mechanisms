import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const artifactModule =
  process.env.ARTIFACT_TOOL_MODULE || "@oai/artifact-tool";
const { Presentation, PresentationFile } = await import(artifactModule);

process.on("beforeExit", () => {
  process.exitCode = 0;
});

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const OUT = path.join(ROOT, "reports", "project_presentation.pptx");
const QA_DIR =
  process.env.PRESENTATION_QA_DIR ||
  path.join(process.env.TEMP || process.env.TMP || ".", "codex-presentations", "explainability-deck", "tmp", "qa");

const C = {
  ink: "#111111",
  muted: "#5C6470",
  light: "#F3F4F6",
  rule: "#C9CED6",
  accent: "#2F80ED",
  accentLight: "#D9ECFF",
  warn: "#E07A2F",
};

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function imageBytes(relativePath) {
  const bytes = await fs.readFile(path.join(ROOT, relativePath));
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: 22,
    color: C.ink,
    fontFace: "Microsoft YaHei",
    ...style,
  };
  return box;
}

function addTitle(slide, title, kicker = "") {
  if (kicker) {
    addText(slide, kicker.toUpperCase(), { left: 64, top: 42, width: 760, height: 26 }, {
      fontSize: 13,
      bold: true,
      color: C.accent,
    });
  }
  addText(slide, title, { left: 64, top: 74, width: 1100, height: 94 }, {
    fontSize: 34,
    bold: true,
    color: C.ink,
  });
  slide.shapes.add({
    geometry: "rect",
    position: { left: 64, top: 176, width: 1120, height: 1 },
    fill: C.rule,
    line: { style: "solid", fill: C.rule, width: 0 },
  });
}

function addFooter(slide, page) {
  addText(slide, "Mechanistic Interpretability Project", { left: 64, top: 664, width: 520, height: 24 }, {
    fontSize: 12,
    color: C.muted,
  });
  addText(slide, String(page).padStart(2, "0"), { left: 1140, top: 664, width: 70, height: 24 }, {
    fontSize: 12,
    color: C.muted,
    alignment: "right",
  });
}

function addBullets(slide, items, left, top, width, fontSize = 21, gap = 44) {
  items.forEach((item, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      position: { left, top: top + i * gap + 8, width: 8, height: 8 },
      fill: C.accent,
      line: { style: "solid", fill: C.accent, width: 0 },
    });
    addText(slide, item, { left: left + 24, top: top + i * gap, width, height: gap - 2 }, {
      fontSize,
      color: C.ink,
    });
  });
}

function addMetric(slide, value, label, left, top, width = 220) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 118 },
    fill: C.light,
    line: { style: "solid", fill: C.rule, width: 1 },
  });
  addText(slide, value, { left: left + 18, top: top + 18, width: width - 36, height: 48 }, {
    fontSize: 34,
    bold: true,
    color: C.accent,
  });
  addText(slide, label, { left: left + 18, top: top + 70, width: width - 36, height: 36 }, {
    fontSize: 15,
    color: C.muted,
  });
}

async function addImage(slide, relativePath, position, alt) {
  slide.images.add({
    blob: await imageBytes(relativePath),
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addCallout(slide, text, left, top, width, height, color = C.accentLight) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
  addText(slide, text, { left: left + 18, top: top + 16, width: width - 36, height: height - 32 }, {
    fontSize: 20,
    bold: true,
    color: C.ink,
  });
}

function addTableText(slide, rows, left, top, colWidths, rowHeight = 44) {
  rows.forEach((row, r) => {
    let x = left;
    row.forEach((cell, c) => {
      slide.shapes.add({
        geometry: "rect",
        position: { left: x, top: top + r * rowHeight, width: colWidths[c], height: rowHeight },
        fill: r === 0 ? "#E8EDF5" : r % 2 === 0 ? "#FFFFFF" : "#F7F8FA",
        line: { style: "solid", fill: "#D6DAE0", width: 1 },
      });
      addText(slide, cell, { left: x + 10, top: top + r * rowHeight + 9, width: colWidths[c] - 20, height: rowHeight - 12 }, {
        fontSize: r === 0 ? 15 : 14,
        bold: r === 0,
        color: C.ink,
      });
      x += colWidths[c];
    });
  });
}

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function newSlide(page, title, kicker = "") {
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  addTitle(slide, title, kicker);
  addFooter(slide, page);
  return slide;
}

// Slide 1
{
  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";
  addText(slide, "机制可解释性课程项目", { left: 64, top: 56, width: 520, height: 28 }, {
    fontSize: 18,
    bold: true,
    color: C.accent,
  });
  addText(slide, "基于 Truth Direction 的\nGPT-2-small 事实判断机制定位", { left: 64, top: 150, width: 840, height: 160 }, {
    fontSize: 52,
    bold: true,
    color: C.ink,
  });
  addText(slide, "Locate / Steer & Improve / Mechanistic Interpretability Reproduction", { left: 68, top: 340, width: 860, height: 38 }, {
    fontSize: 22,
    color: C.muted,
  });
  addMetric(slide, "528", "true/false factual statements", 68, 470, 220);
  addMetric(slide, "0.953", "best capital probe AUC", 318, 470, 240);
  addMetric(slide, "3", "intervention families", 588, 470, 220);
  addCallout(slide, "核心结论：可读的 truth direction 存在，但可读性不等于直接可控性。", 850, 448, 330, 150);
  addFooter(slide, 1);
}

// Slide 2
{
  const slide = newSlide(2, "研究问题聚焦在一个可复现实验闭环", "Question");
  addBullets(slide, [
    "哪些层能线性读出 true/false 信息？",
    "这种信息是否跨事实领域稳定？",
    "学到的 direction 能否被 steering 或 ablation 控制？",
    "Activation patching 能否提供更强因果证据？",
  ], 88, 206, 650, 24, 62);
  addCallout(slide, "实验闭环：定位表示 → 可视化结构 → 干预方向 → 分析负结果", 790, 220, 350, 180);
}

// Slide 3
{
  const slide = newSlide(3, "数据集设计控制泄漏，并保留跨领域比较", "Dataset");
  addMetric(slide, "264 / 264", "true / false balanced", 72, 190, 270);
  addMetric(slide, "7", "fact domains", 372, 190, 200);
  addMetric(slide, "group split", "same pair_id stays in one split", 602, 190, 280);
  addTableText(slide, [
    ["Domain", "Rows"],
    ["capital", "152"],
    ["continent", "86"],
    ["element_symbol", "80"],
    ["book_author", "60"],
    ["landmark_country", "60"],
    ["science + math", "90"],
  ], 84, 358, [360, 160], 38);
  addBullets(slide, [
    "模型：GPT-2-small",
    "框架：TransformerLens",
    "Hook：resid_post / attn_out / mlp_out",
  ], 690, 380, 420, 20, 46);
}

// Slide 4
{
  const slide = newSlide(4, "Truth signal is strongest in structured capital facts", "Locate");
  await addImage(slide, "figures/probe_sweep_summary.png", { left: 70, top: 180, width: 720, height: 410 }, "Probe sweep summary chart");
  addCallout(slide, "Capital + answer prompt reaches AUC 0.953 at layer 8.", 840, 210, 300, 110);
  addBullets(slide, [
    "混合领域信号被任务差异稀释。",
    "truth direction 更像领域相关结构。",
    "后续主实验聚焦 capital。",
  ], 835, 360, 330, 18, 48);
}

// Slide 5
{
  const slide = newSlide(5, "High-dimensional probe succeeds where PCA does not fully separate", "Focused Probe");
  await addImage(slide, "figures/probe_capital_answer.png", { left: 62, top: 182, width: 540, height: 360 }, "Capital probe layer curve");
  await addImage(slide, "figures/pca_capital_layer8.png", { left: 660, top: 182, width: 520, height: 360 }, "Layer 8 PCA scatter plot");
  addCallout(slide, "Best AUC: layer 8 = 0.953. Best accuracy: layer 10 = 0.870.", 110, 570, 980, 62);
}

// Slide 6
{
  const slide = newSlide(6, "Errors show probe readability is not a perfect fact checker", "Error Analysis");
  addMetric(slide, "46", "test samples", 78, 178, 180);
  addMetric(slide, "38", "correct", 278, 178, 180);
  addMetric(slide, "8", "errors", 478, 178, 180);
  addTableText(slide, [
    ["Statement", "Label", "Pred."],
    ["Laos is Vientiane", "true", "false"],
    ["Canada is Amman", "false", "true"],
    ["Chile is Santiago", "true", "false"],
    ["India is New Delhi", "true", "false"],
    ["Nigeria is Mexico City", "false", "true"],
  ], 76, 348, [520, 120, 120], 40);
  addCallout(slide, "AUC 高说明排序能力强；固定 0.5 阈值仍会产生偏置。", 830, 335, 300, 150, "#FFF1D6");
}

// Slide 7
{
  const slide = newSlide(7, "Activation patching gives causal evidence in capital recall", "Causal Locate");
  await addImage(slide, "figures/activation_patching_capital_recall.png", { left: 70, top: 180, width: 740, height: 420 }, "Activation patching recovery by layer");
  addBullets(slide, [
    "resid_post layer 11 recovery = 1.000",
    "attn_out contributes strongly but is not sufficient alone",
    "mlp_out patching is weaker",
  ], 850, 255, 330, 19, 54);
}

// Slide 8
{
  const slide = newSlide(8, "Steering moves the internal score, ablation reveals redundancy", "Steer & Improve");
  await addImage(slide, "figures/steering_capital_probe_layer8_probe_accuracy.png", { left: 62, top: 182, width: 520, height: 360 }, "Steering probe accuracy curve");
  await addImage(slide, "figures/ablation_capital_probe_layer8_score_gap.png", { left: 660, top: 182, width: 520, height: 360 }, "Ablation score gap curve");
  addCallout(slide, "负结果也重要：global steering 未提升输出判断；单方向 ablation 后重新训练 probe 仍有高 AUC。", 92, 570, 1060, 64, "#E8F4FF");
}

// Slide 9
{
  const slide = newSlide(9, "This is a small-scale reproduction plus domain-stability analysis", "Reproduction");
  addTableText(slide, [
    ["Paper idea", "Project evidence"],
    ["True/false datasets", "528 balanced factual statements"],
    ["Linear truth representation", "Layer-wise residual stream probe"],
    ["Visualization", "PCA activation scatter"],
    ["Causal intervention", "Patching, steering, ablation"],
    ["Generalization question", "Domain and prompt sweep"],
  ], 82, 186, [410, 650], 56);
  addCallout(slide, "拓展点：报告不只展示成功，也明确保留跨领域弱泛化与 steering 负结果。", 170, 560, 900, 62);
}

// Slide 10
{
  const slide = newSlide(10, "Takeaway: truth direction is useful, but not a universal button", "Conclusion");
  addBullets(slide, [
    "Capital fact verification 中存在强线性 truth/false 表征。",
    "后层 residual stream 对 capital recall 有清晰因果作用。",
    "可读性不等于直接可控性；naive steering 不足以 improve 输出。",
    "单一 direction 之外仍有冗余子空间，后续可做 head-level patching 与多方向 ablation。",
  ], 100, 190, 900, 25, 76);
  addCallout(slide, "最终观点：机制可解释性需要同时报告定位成功、因果证据、失败样本和负结果。", 120, 560, 960, 70);
}

await fs.mkdir(path.dirname(OUT), { recursive: true });
await fs.mkdir(QA_DIR, { recursive: true });

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(QA_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf-8");
}

await writeBlob(path.join(QA_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT);
await fs.rm(`${OUT}.inspect.ndjson`, { force: true });
console.log(`Saved deck to ${OUT}`);
console.log(`QA previews saved to ${QA_DIR}`);
process.exit(0);
