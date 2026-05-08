# Research Studio UI Refinement Plan

## Objective
Enhance the Research Studio (v2.0) interface to improve visual clarity, clarify the three-phase workflow (Discover -> Validate -> Promote), and eliminate the "wall of text" feel. The goal is to make the tool feel like a professional cockpit for strategic content decision-making.

## 🎨 Design Language
*   **Phase 1 (Discovery)**: **Indigo** (#6366f1) - The starting point.
*   **Phase 2 (Validation)**: **Purple** (#8b5cf6) - The engine room.
*   **Phase 3 (Promotion)**: **Emerald** (#10b981) - The winning outcome.

---

## 🏗️ Structural Improvements

### 1. Panel Distinction
*   **Left Panel (Discovery)**: Use a subtle border and slightly darker background (`#0d0d0f`) to distinguish it as the "Input/Control" area.
*   **Right Panel (Analysis)**: Use a "Glassmorphism" effect with a very subtle inner glow to denote it as the "Output/Intelligence" area.
*   **Visual Connectors**: Add a vertical "flow line" that connects the Job Node to its child Opportunity Cards.

### 2. Phase-Driven UX
*   **Active Phase Highlighting**: Dim inactive panels or add a subtle glow to the section headers of the current focus.
*   **Dynamic CTAs**: Use pulsing animations or distinct colors for the "Primary Action" in each phase (e.g., "Generate Jobs", "Initiate Validation", "Promote Idea").

---

## 🃏 OpportunityCard Refinement

### 1. Header Hierarchy
*   **Title**: Increase font size and use `font-black` for the primary opportunity text.
*   **Route Badge**: Move the Route Badge to the top-right and give it a "Stamp" or "Certified" feel.
*   **Compact Metrics**: Show only the primary "Efficiency" score in the header; hide sub-scores (Intent, SERP, Gap) until inspection.

### 2. Analysis Dashboard (Expanded View)
*   **Grid Layout**: Use a 3-column grid for metrics (Search Potential, Competition, Feasibility).
*   **Visual Evidence**: Use actual screenshot/favicon previews for SERP evidence instead of just text blocks.
*   **Actionable Feedback**: Add a "Why this won" or "Why this failed" natural language summary (Reason Codes).

---

## 🛠️ Implementation Tasks

### High Priority
- [ ] **Refactor Section Headers**: Add Phase Numbers and clear descriptions to each panel.
- [ ] **Add "Next Step" Indicators**: Visual cues (e.g., arrows or pulsing buttons) that guide the user from Discover to Validate.
- [ ] **Improve Empty States**: Create interactive placeholders that explain *how* to populate the current view.
- [ ] **Enhance Typography**: Increase contrast and use consistent sizing to guide the eye (Hierarchy).

### Medium Priority
- [ ] **Sticky Control Bar**: Ensure the filter/search bar in the Right Panel stays visible during long scrolls.
- [ ] **Hover Interactions**: Add subtle scale/glow effects to cards when hovered.
- [ ] **Transition Animations**: Use `framer-motion` (or standard CSS transitions) for smooth card expansion and status updates.

### Polish
- [ ] **Share Link Preview**: Add a small toast notification that shows a preview of the shared URL.
- [ ] **Dark Mode Optimization**: Ensure text readability across all background variations.

---

## ✅ Success Criteria
*   User can identify the "Next Step" in under 2 seconds.
*   The difference between a "Discovery Job" and a "Validated Opportunity" is visually obvious.
*   The interface feels "alive" and responsive to user actions.
