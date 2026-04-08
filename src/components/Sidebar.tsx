import { useState } from 'react';
import type { Book, ColourMode, ViewMode, InterpolateState, Metadata } from '../utils/types';
import { BOOK_COLOURS } from '../utils/colours';
import { COLOUR_MODE_INFO, VIEW_MODE_INFO, FIELD_OVERLAY_INFO } from '../utils/displaySettings';
import { ColourLegend } from './ColourLegend';
import { PlaybackControls } from './PlaybackControls';
import { PersistenceGlyph } from './PersistenceGlyph';

interface Props {
  books: Book[];
  metadata: Metadata | null;
  activeBooks: Set<number>;
  colourMode: ColourMode;
  viewMode: ViewMode;
  fieldEnabled: boolean;
  interpolate: InterpolateState;
  onToggleBook: (index: number) => void;
  onSetAllBooks: (active: boolean) => void;
  onSetColourMode: (mode: ColourMode) => void;
  onSetViewMode: (mode: ViewMode) => void;
  onSetFieldEnabled: (enabled: boolean) => void;
  onSetInterpolate: (state: InterpolateState) => void;
  singleActiveBook: number | null;
  hoveredBookIndex: number | null;
  onHoverBook: (index: number | null) => void;
  playbackState: {
    playing: boolean;
    currentIndex: number;
    speed: number;
    togglePlay: () => void;
    setCurrentIndex: (i: number) => void;
    setSpeed: (s: number) => void;
    reset: () => void;
  } | null;
}

const COLOUR_MODE_ORDER: ColourMode[] = [
  'book', 'dialogue', 'entropy', 'sent_len', 'position',
  'motif', 'surprise', 'shape',
  'spec1', 'spec2', 'spec3', 'spec4',
];

const VIEW_MODE_ORDER: ViewMode[] = ['default', 'canonical', 'interpolate'];

/** Reusable inline blurb shown under each setting group. */
function SettingBlurb({ name, blurb }: { name: string; blurb: string }) {
  return (
    <div className="px-4 pb-3 text-xs text-[#6b5c4d] leading-snug">
      <div className="font-semibold text-[#3d3328] mb-0.5">{name}</div>
      <div className="italic text-[#8a7e6b]">{blurb}</div>
    </div>
  );
}

export function Sidebar({
  books,
  metadata,
  activeBooks,
  colourMode,
  viewMode,
  fieldEnabled,
  interpolate,
  onToggleBook,
  onSetAllBooks,
  onSetColourMode,
  onSetViewMode,
  onSetFieldEnabled,
  onSetInterpolate,
  singleActiveBook,
  hoveredBookIndex,
  onHoverBook,
  playbackState,
}: Props) {
  const [search, setSearch] = useState('');
  const filtered = books
    .map((b, i) => ({ book: b, index: i }))
    .filter(({ book }) => book.title.toLowerCase().includes(search.toLowerCase()));

  const hasField = !!metadata?.field;
  const isInterp = viewMode === 'interpolate';

  // In interpolate mode, clicking a book picks slot A then slot B (then resets).
  const handleInterpClick = (idx: number) => {
    if (interpolate.bookA === null) {
      onSetInterpolate({ ...interpolate, bookA: idx });
    } else if (interpolate.bookB === null && idx !== interpolate.bookA) {
      onSetInterpolate({ ...interpolate, bookB: idx });
    } else {
      onSetInterpolate({ ...interpolate, bookA: idx, bookB: null });
    }
  };

  return (
    <div className="w-[300px] h-screen bg-[#ece5d8] border-l border-[#d4cbb8] flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-1">
        <h1 className="text-lg font-semibold text-[#3d3328] tracking-wide">Shapes of Stories</h1>
        <p className="text-xs text-[#8a7e6b] italic mt-0.5">Narrative space explorer</p>
      </div>

      {/* View mode */}
      <div className="px-4 pt-3 pb-1">
        <div className="text-xs text-[#8a7e6b] mb-1.5 uppercase tracking-widest" style={{ fontSize: '0.65rem' }}>Projection</div>
        <div className="flex gap-1">
          {VIEW_MODE_ORDER.map((value) => (
            <button
              key={value}
              onClick={() => onSetViewMode(value)}
              className={`flex-1 px-2 py-1 text-xs rounded-sm transition-colors ${
                viewMode === value
                  ? 'bg-[#3d3328] text-[#f5f0e8]'
                  : 'bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8]'
              }`}
            >
              {VIEW_MODE_INFO[value].label}
            </button>
          ))}
        </div>
      </div>
      <SettingBlurb
        name={VIEW_MODE_INFO[viewMode].name}
        blurb={VIEW_MODE_INFO[viewMode].blurb}
      />

      {/* Field toggle (only when E04 data is present) */}
      {hasField && (
        <>
          <div className="px-4 pt-1 pb-1">
            <label className="flex items-center gap-2 text-xs text-[#6b5c4d] cursor-pointer">
              <input
                type="checkbox"
                checked={fieldEnabled}
                onChange={(e) => onSetFieldEnabled(e.target.checked)}
                className="accent-[#3d3328]"
              />
              <span>{FIELD_OVERLAY_INFO.label}</span>
            </label>
          </div>
          {fieldEnabled && (
            <SettingBlurb
              name={FIELD_OVERLAY_INFO.name}
              blurb={FIELD_OVERLAY_INFO.blurb}
            />
          )}
        </>
      )}

      {/* Colour mode */}
      <div className="px-4 pt-3 pb-2">
        <div className="text-xs text-[#8a7e6b] mb-1.5 uppercase tracking-widest" style={{ fontSize: '0.65rem' }}>Colour by</div>
        <div className="flex flex-wrap gap-1">
          {COLOUR_MODE_ORDER.map((value) => (
            <button
              key={value}
              onClick={() => onSetColourMode(value)}
              className={`px-2.5 py-1 text-xs rounded-sm transition-colors ${
                colourMode === value
                  ? 'bg-[#3d3328] text-[#f5f0e8]'
                  : 'bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8]'
              }`}
            >
              {COLOUR_MODE_INFO[value].label}
            </button>
          ))}
        </div>
      </div>

      <ColourLegend
        mode={colourMode}
        vqK={metadata?.vq_k}
        shapeArchetypes={metadata?.shape_archetypes}
        diffusionEigenvalues={metadata?.diffusion?.eigenvalues}
      />

      <SettingBlurb
        name={COLOUR_MODE_INFO[colourMode].name}
        blurb={COLOUR_MODE_INFO[colourMode].blurb}
      />

      {/* E08 + E09 — global spectral metrics */}
      {(metadata?.transition_svd || metadata?.velocity_pca) && (
        <div className="px-4 pb-3">
          <div className="text-xs text-[#8a7e6b] mb-1 uppercase tracking-widest" style={{ fontSize: '0.6rem' }}>
            Spectral metrics
          </div>
          {metadata?.transition_svd && (
            <div className="text-xs text-[#6b5c4d] tabular-nums leading-tight mb-1">
              <span className="text-[#8a7e6b]">Linear T:</span>{' '}
              σ = [{metadata.transition_svd.sigma.map((s) => s.toFixed(2)).join(', ')}],{' '}
              R² = {metadata.transition_svd.r2.toFixed(2)}
            </div>
          )}
          {metadata?.velocity_pca && (
            <div className="text-xs text-[#6b5c4d] tabular-nums leading-tight">
              <span className="text-[#8a7e6b]">Velocity PCA:</span>{' '}
              {metadata.velocity_pca.variance_ratio
                .map((v) => `${(v * 100).toFixed(0)}%`)
                .join(' / ')}
            </div>
          )}
        </div>
      )}

      {/* Interpolate panel */}
      {isInterp && (
        <div className="px-4 pb-2">
          <div className="text-xs text-[#8a7e6b] mb-1 italic">
            {interpolate.bookA === null
              ? 'Click a book below to pick A'
              : interpolate.bookB === null
              ? 'Pick B (or click A again to reset)'
              : 'Drag the slider to interpolate'}
          </div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-[#6b5c4d] w-4">A</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={interpolate.alpha}
              onChange={(e) => onSetInterpolate({ ...interpolate, alpha: Number(e.target.value) })}
              disabled={interpolate.bookA === null || interpolate.bookB === null}
              className="flex-1 accent-[#3d3328]"
            />
            <span className="text-xs text-[#6b5c4d] w-4">B</span>
          </div>
          <div className="text-xs text-[#8a7e6b] text-center tabular-nums mb-2">
            α = {interpolate.alpha.toFixed(2)}
          </div>
          {hasField && (
            <>
              <div className="text-[10px] text-[#8a7e6b] uppercase tracking-widest mb-1">Field influence</div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={interpolate.fieldInfluence}
                onChange={(e) => onSetInterpolate({ ...interpolate, fieldInfluence: Number(e.target.value) })}
                disabled={interpolate.bookA === null || interpolate.bookB === null}
                className="w-full accent-[#3d3328]"
              />
              <div className="text-xs text-[#8a7e6b] text-center tabular-nums">
                {interpolate.fieldInfluence === 0
                  ? 'pure lerp'
                  : `bent by v(z) × ${interpolate.fieldInfluence.toFixed(2)}`}
              </div>
            </>
          )}
        </div>
      )}

      {/* Search + bulk buttons */}
      <div className="px-4 pb-2">
        <input
          type="text"
          placeholder="Search books..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-2.5 py-1.5 text-sm bg-[#f5f0e8] text-[#3d3328] rounded-sm border border-[#d4cbb8] focus:outline-none focus:border-[#a89880] placeholder-[#b5a992] mb-2"
        />
        <div className="flex gap-2">
          <button
            onClick={() => onSetAllBooks(true)}
            className="flex-1 px-2 py-1 text-xs rounded-sm bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8] transition-colors"
          >
            All on
          </button>
          <button
            onClick={() => onSetAllBooks(false)}
            className="flex-1 px-2 py-1 text-xs rounded-sm bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8] transition-colors"
          >
            All off
          </button>
        </div>
      </div>

      {/* Book list */}
      <div
        className="flex-1 overflow-y-auto px-4 pb-2"
        onMouseLeave={() => onHoverBook(null)}
      >
        {filtered.map(({ book, index }) => {
          const colour = BOOK_COLOURS[index % BOOK_COLOURS.length];
          const active = activeBooks.has(index);
          const isInterpA = isInterp && interpolate.bookA === index;
          const isInterpB = isInterp && interpolate.bookB === index;
          const hoverDimmed = hoveredBookIndex !== null && hoveredBookIndex !== index;
          const selectionDimmed = !isInterp && activeBooks.size > 0 && !active && hoveredBookIndex === null;
          const opacity = hoverDimmed ? 0.3 : selectionDimmed ? 0.5 : 1;
          const highlighted = active || isInterpA || isInterpB;
          return (
            <button
              key={index}
              onClick={() => (isInterp ? handleInterpClick(index) : onToggleBook(index))}
              onMouseEnter={() => onHoverBook(index)}
              className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-sm text-left text-sm mb-0.5 transition-all duration-150 ${
                highlighted
                  ? 'bg-[#f5f0e8] text-[#3d3328]'
                  : 'text-[#8a7e6b] hover:text-[#5a4d3e] hover:bg-[#e8e0d0]'
              }`}
              style={{ opacity }}
            >
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{
                  backgroundColor: highlighted ? colour : 'transparent',
                  border: `2px solid ${colour}`,
                }}
              />
              <span className="truncate flex-1">{book.title}</span>
              {isInterpA && <span className="text-[10px] text-[#3d3328] font-semibold">A</span>}
              {isInterpB && <span className="text-[10px] text-[#3d3328] font-semibold">B</span>}
              <PersistenceGlyph pairs={book.persistence} colour={colour} />
            </button>
          );
        })}
      </div>

      {/* Playback controls */}
      {singleActiveBook !== null && playbackState && (
        <PlaybackControls
          playing={playbackState.playing}
          currentIndex={playbackState.currentIndex}
          total={books[singleActiveBook].passages.length}
          speed={playbackState.speed}
          onTogglePlay={playbackState.togglePlay}
          onSetIndex={playbackState.setCurrentIndex}
          onSetSpeed={playbackState.setSpeed}
          onReset={playbackState.reset}
        />
      )}
    </div>
  );
}
