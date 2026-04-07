// Richer, deeper palette suited for light/cream backgrounds
export const BOOK_COLOURS = [
  '#C0392B', '#2874A6', '#1E8449', '#884EA0', '#CA6F1E',
  '#17A589', '#6C3483', '#2E86C1', '#D4AC0D', '#A93226',
  '#148F77', '#7D3C98', '#D68910', '#1A5276', '#B03A2E',
  '#239B56', '#AF601A', '#2471A3', '#76448A', '#1ABC9C',
  '#E74C3C', '#3498DB', '#27AE60', '#8E44AD', '#F39C12',
  '#16A085', '#9B59B6', '#2980B9', '#D35400', '#138D75',
];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function rgbToHex(r: number, g: number, b: number): string {
  return '#' + ((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1);
}

function gradientColour(stops: string[], t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const segment = clamped * (stops.length - 1);
  const i = Math.min(Math.floor(segment), stops.length - 2);
  const frac = segment - i;
  const [r1, g1, b1] = hexToRgb(stops[i]);
  const [r2, g2, b2] = hexToRgb(stops[i + 1]);
  return rgbToHex(
    Math.round(lerp(r1, r2, frac)),
    Math.round(lerp(g1, g2, frac)),
    Math.round(lerp(b1, b2, frac))
  );
}

// Dialogue: amber → vermilion
export function dialogueColour(v: number): string {
  return gradientColour(['#F5DEB3', '#D4AC0D', '#C0392B'], v);
}

// Entropy: deep indigo → gold
export function entropyColour(v: number): string {
  const t = Math.max(0, Math.min(1, (v - 1) / 5));
  return gradientColour(['#1A237E', '#7B1FA2', '#D4AC0D'], t);
}

// Sentence length: forest → sage
export function sentLenColour(v: number): string {
  const t = Math.max(0, Math.min(1, (v - 5) / 40));
  return gradientColour(['#0D3B2E', '#2E7D32', '#81C784'], t);
}

// Position: slate blue → burgundy
export function positionColour(v: number): string {
  return gradientColour(['#1A5276', '#7B1FA2', '#C0392B'], v);
}

// E03 — motif: distinct hues per code (max 16)
const MOTIF_PALETTE = [
  '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C',
  '#E67E22', '#34495E', '#16A085', '#C0392B', '#2980B9', '#27AE60',
  '#D35400', '#8E44AD', '#7F8C8D', '#F1C40F',
];
export function motifColour(code: number | undefined): string {
  if (code === undefined || code < 0) return '#aaa';
  return MOTIF_PALETTE[code % MOTIF_PALETTE.length];
}

// E04 — surprise: pale → magma red
export function surpriseColour(v: number, max: number = 3): string {
  const t = Math.max(0, Math.min(1, v / max));
  return gradientColour(['#f5e6d3', '#E67E22', '#C0392B', '#3d1818'], t);
}

// E04 — vector field magnitude: deep navy → teal → green → amber → vermilion.
// Roughly perceptually uniform across t ∈ [0, 1].
const MAGNITUDE_STOPS = ['#1a1a4e', '#1f4d6e', '#138a72', '#9bc53d', '#f4a261', '#e76f51'];
export function magnitudeColour(t: number): [number, number, number] {
  const hex = gradientColour(MAGNITUDE_STOPS, Math.max(0, Math.min(1, t)));
  const [r, g, b] = hexToRgb(hex);
  return [r / 255, g / 255, b / 255];
}

// E02 — shape archetype: distinct hues per cluster
const SHAPE_PALETTE = [
  '#1A5276', '#C0392B', '#0D3B2E', '#7B1FA2', '#D4AC0D', '#16A085',
  '#884EA0', '#CA6F1E',
];
export function shapeColour(archetype: number | undefined): string {
  if (archetype === undefined || archetype < 0) return '#aaa';
  return SHAPE_PALETTE[archetype % SHAPE_PALETTE.length];
}

interface FeatureSource {
  dialogue: number;
  entropy: number;
  sent_len: number;
  pos: number;
  code?: number;
  surprise?: number;
}

export function getFeatureColour(mode: string, passage: FeatureSource, book?: { shape_archetype?: number }): string {
  switch (mode) {
    case 'dialogue': return dialogueColour(passage.dialogue);
    case 'entropy': return entropyColour(passage.entropy);
    case 'sent_len': return sentLenColour(passage.sent_len);
    case 'position': return positionColour(passage.pos);
    case 'motif': return motifColour(passage.code);
    case 'surprise': return surpriseColour(passage.surprise ?? 0);
    case 'shape': return shapeColour(book?.shape_archetype);
    default: return '#888';
  }
}
