# Research Studio UI Refinement Plan (Updated)

## Objective
Transform the Research Studio from a data-heavy display into a guided, intuitive cockpit. The user should feel mentored through the three phases: **Discover**, **Validate**, and **Promote**.

## 🎨 Phase Visual Identities
*   **Phase 1 (Discovery)**: **Indigo** (#6366f1) -> "Input / Intent"
*   **Phase 2 (Validation)**: **Purple** (#8b5cf6) -> "The Engine / Analysis"
*   **Phase 3 (Intelligence)**: **Emerald** (#10b981) -> "Outcomes / ROI"

---

## 🏗️ Structural & Instructional Overhaul

### 1. Panel Distinction & Ambient Glow
*   **Phase 1 Backdrop**: Use a subtle Indigo tint (`bg-indigo-500/[0.02]`) for the left column.
*   **Phase 3 Backdrop**: Use a subtle Emerald tint (`bg-emerald-500/[0.02]`) for the right column.
*   **Section Glows**: Add "Atmospheric Glow" divs behind section headers to anchor the eye.

### 2. Guided "Quick-Start" Cards
*   Each phase should start with a small, dismissible "Phase Guide" card that says:
    *   **Phase 1**: "Define your niche. Generate or add 'User Jobs' (what people want to do) to start."
    *   **Phase 2**: "Validation is the core. Approve jobs to move them into the validation queue."
    *   **Phase 3**: "Promote winning opportunities to Content Studio or Release Software."

### 3. "Wall of Text" Mitigation
*   **Typography**: Use larger, bolder headings and more white space (or dark space).
*   **Visual Separators**: Use color-coded horizontal "Phase Dividers".
*   **Metric Icons**: Every metric (Intent, SERP, Gap, Ease) must have a tooltip explaining *why* it matters.

---

## 🃏 OpportunityCard (Outcome) Refinement

### 1. The "Decision Stamp"
*   Add a large, translucent "Outcome Type" stamp in the background of the card (e.g., "ARTICLE", "SOFTWARE").

### 2. Competitive Visuals
*   Instead of just a URL, show a "Competitor Card" with the favicon and a domain authority badge (mocked if data is missing).

---

## 🛠️ Implementation Tasks

### High Priority (Instructional UX)
- [ ] **Phase Guide Cards**: Add instructional cards at the top of each section.
- [ ] **Section Background Tints**: Apply color-coded backgrounds to left and right panels.
- [ ] **Instructional Empty States**: Replace "No jobs found" with "Step 1: Select a Category above to generate jobs."
- [ ] **Next Step Pulse**: Add a pulsing glow to the "Generate" or "Validate" buttons if the user is stuck.

### UI Polish
- [ ] **Flow Connectors**: Make the Indigo "flow line" more prominent and glowy.
- [ ] **Score Tooltips**: Add informational hover states to the mini-meters.
- [ ] **Decision Stamps**: Background watermark for outcome types.

---

## ✅ Success Criteria
*   User knows exactly where to click next without reading more than 5 words.
*   The transition from "Input" to "Output" is visually dramatic (Indigo -> Emerald).
*   The "Wall of Text" is broken by distinct visual blocks and color anchors.

