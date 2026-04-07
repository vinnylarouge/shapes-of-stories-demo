import type { PersistencePair } from '../utils/types';

interface Props {
  pairs?: PersistencePair[];
  size?: number;
  colour?: string;
}

/** Tiny per-book persistence glyph for the sidebar. Renders the H₀ death
 * times as marks along a vertical line; longer marks = more persistent
 * components. Falls back to nothing if persistence isn't loaded. */
export function PersistenceGlyph({ pairs, size = 14, colour = '#8a7e6b' }: Props) {
  if (!pairs || pairs.length === 0) {
    return <span className="inline-block" style={{ width: size, height: size }} />;
  }

  // Normalise deaths to [0, 1] within this book.
  const deaths = pairs.map((p) => p.death);
  const max = Math.max(...deaths, 1e-6);
  const w = size;
  const h = size;
  const baseY = h - 1;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="flex-shrink-0">
      {/* baseline */}
      <line x1={0.5} y1={baseY} x2={w - 0.5} y2={baseY} stroke={colour} strokeWidth={0.6} opacity={0.4} />
      {deaths.map((d, i) => {
        const x = ((i + 0.5) / deaths.length) * w;
        const lh = (d / max) * (h - 2);
        return (
          <line
            key={i}
            x1={x}
            x2={x}
            y1={baseY}
            y2={baseY - lh}
            stroke={colour}
            strokeWidth={1.2}
            strokeLinecap="round"
            opacity={0.9}
          />
        );
      })}
    </svg>
  );
}
