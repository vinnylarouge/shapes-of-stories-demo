import type { ColourMode } from '../utils/types';
import {
  dialogueColour,
  entropyColour,
  sentLenColour,
  positionColour,
  surpriseColour,
  motifColour,
  shapeColour,
  eigColour,
} from '../utils/colours';

interface Props {
  mode: ColourMode;
  vqK?: number;
  shapeArchetypes?: number;
  diffusionEigenvalues?: number[];
}

const GRADIENT_LEGENDS: Record<string, { label: string; fn: (t: number) => string; min: string; max: string }> = {
  dialogue: { label: 'Dialogue Fraction', fn: dialogueColour, min: '0', max: '1' },
  entropy: { label: 'Entropy', fn: (t) => entropyColour(1 + t * 5), min: '1', max: '6' },
  sent_len: { label: 'Sentence Length', fn: (t) => sentLenColour(5 + t * 40), min: '5', max: '45' },
  position: { label: 'Position in Book', fn: positionColour, min: '0%', max: '100%' },
  surprise: { label: 'Surprise (vs. field)', fn: (t) => surpriseColour(t * 3), min: '0', max: '3+' },
};

const SPEC_MODE_INDEX: Record<string, number> = { spec1: 0, spec2: 1, spec3: 2, spec4: 3 };

export function ColourLegend({ mode, vqK, shapeArchetypes, diffusionEigenvalues }: Props) {
  if (mode === 'book') return null;

  if (mode === 'motif' || mode === 'shape') {
    const k = mode === 'motif' ? (vqK ?? 8) : (shapeArchetypes ?? 4);
    const fn = mode === 'motif' ? motifColour : shapeColour;
    const label = mode === 'motif' ? 'Motif Code' : 'Shape Archetype';
    return (
      <div className="px-4 pb-3">
        <div className="text-xs text-[#8a7e6b] mb-1 italic">{label}</div>
        <div className="flex flex-wrap gap-1">
          {Array.from({ length: k }, (_, i) => (
            <div key={i} className="flex items-center gap-1">
              <span
                className="w-2.5 h-2.5 rounded-sm inline-block"
                style={{ backgroundColor: fn(i) }}
              />
              <span className="text-[10px] text-[#8a7e6b]">{i}</span>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (mode in SPEC_MODE_INDEX) {
    const idx = SPEC_MODE_INDEX[mode];
    const eigval = diffusionEigenvalues?.[idx];
    const stops = Array.from({ length: 20 }, (_, i) => {
      const t = i / 19;
      // Map t ∈ [0,1] back to v ∈ [-0.05, +0.05]
      return eigColour((t - 0.5) * 0.1);
    });
    const gradient = `linear-gradient(to right, ${stops.join(', ')})`;
    return (
      <div className="px-4 pb-3">
        <div className="text-xs text-[#8a7e6b] mb-1 italic">
          Diffusion eigenvector {idx + 1}
          {eigval !== undefined && (
            <span className="text-[#a89880] tabular-nums"> (λ={eigval.toFixed(4)})</span>
          )}
        </div>
        <div className="h-2.5 rounded-sm" style={{ background: gradient }} />
        <div className="flex justify-between text-xs text-[#a89880] mt-0.5">
          <span>−</span>
          <span>0</span>
          <span>+</span>
        </div>
      </div>
    );
  }

  const legend = GRADIENT_LEGENDS[mode];
  if (!legend) return null;

  const stops = Array.from({ length: 20 }, (_, i) => {
    const t = i / 19;
    return legend.fn(t);
  });
  const gradient = `linear-gradient(to right, ${stops.join(', ')})`;

  return (
    <div className="px-4 pb-3">
      <div className="text-xs text-[#8a7e6b] mb-1 italic">{legend.label}</div>
      <div className="h-2.5 rounded-sm" style={{ background: gradient }} />
      <div className="flex justify-between text-xs text-[#a89880] mt-0.5">
        <span>{legend.min}</span>
        <span>{legend.max}</span>
      </div>
    </div>
  );
}
