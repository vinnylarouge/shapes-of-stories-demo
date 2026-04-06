import { useState } from 'react';
import type { Book, ColourMode } from '../utils/types';
import { BOOK_COLOURS } from '../utils/colours';
import { ColourLegend } from './ColourLegend';
import { PlaybackControls } from './PlaybackControls';

interface Props {
  books: Book[];
  activeBooks: Set<number>;
  colourMode: ColourMode;
  onToggleBook: (index: number) => void;
  onSetAllBooks: (active: boolean) => void;
  onSetColourMode: (mode: ColourMode) => void;
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

const COLOUR_MODES: { value: ColourMode; label: string }[] = [
  { value: 'book', label: 'By Book' },
  { value: 'dialogue', label: 'Dialogue' },
  { value: 'entropy', label: 'Entropy' },
  { value: 'sent_len', label: 'Sentence Len.' },
  { value: 'position', label: 'Position' },
];

export function Sidebar({
  books,
  activeBooks,
  colourMode,
  onToggleBook,
  onSetAllBooks,
  onSetColourMode,
  singleActiveBook,
  hoveredBookIndex,
  onHoverBook,
  playbackState,
}: Props) {
  const [search, setSearch] = useState('');
  const filtered = books
    .map((b, i) => ({ book: b, index: i }))
    .filter(({ book }) => book.title.toLowerCase().includes(search.toLowerCase()));

  return (
    <div className="w-[300px] h-screen bg-[#ece5d8] border-l border-[#d4cbb8] flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-4 pt-4 pb-1">
        <h1 className="text-lg font-semibold text-[#3d3328] tracking-wide">Shapes of Stories</h1>
        <p className="text-xs text-[#8a7e6b] italic mt-0.5">Narrative space explorer</p>
      </div>

      {/* Colour mode */}
      <div className="px-4 pt-3 pb-2">
        <div className="text-xs text-[#8a7e6b] mb-1.5 uppercase tracking-widest" style={{ fontSize: '0.65rem' }}>Colouring</div>
        <div className="flex flex-wrap gap-1">
          {COLOUR_MODES.map((m) => (
            <button
              key={m.value}
              onClick={() => onSetColourMode(m.value)}
              className={`px-2.5 py-1 text-xs rounded-sm transition-colors ${
                colourMode === m.value
                  ? 'bg-[#3d3328] text-[#f5f0e8]'
                  : 'bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8]'
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <ColourLegend mode={colourMode} />

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
          const hoverDimmed = hoveredBookIndex !== null && hoveredBookIndex !== index;
          const selectionDimmed = activeBooks.size > 0 && !active && hoveredBookIndex === null;
          const opacity = hoverDimmed ? 0.3 : selectionDimmed ? 0.5 : 1;
          return (
            <button
              key={index}
              onClick={() => onToggleBook(index)}
              onMouseEnter={() => onHoverBook(index)}
              className={`w-full flex items-center gap-2.5 px-2 py-1.5 rounded-sm text-left text-sm mb-0.5 transition-all duration-150 ${
                active
                  ? 'bg-[#f5f0e8] text-[#3d3328]'
                  : 'text-[#8a7e6b] hover:text-[#5a4d3e] hover:bg-[#e8e0d0]'
              }`}
              style={{ opacity }}
            >
              <span
                className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                style={{
                  backgroundColor: active ? colour : 'transparent',
                  border: `2px solid ${colour}`,
                }}
              />
              <span className="truncate">{book.title}</span>
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
