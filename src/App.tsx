import { useState, useEffect, useCallback, useMemo } from 'react';
import type { StoryData, ColourMode, HoveredPassage } from './utils/types';
import { Canvas } from './components/Canvas';
import { Sidebar } from './components/Sidebar';
import { PassagePanel } from './components/PassagePanel';
import { usePlayback } from './hooks/usePlayback';

export default function App() {
  const [data, setData] = useState<StoryData | null>(null);
  const [activeBooks, setActiveBooks] = useState<Set<number>>(new Set());
  const [colourMode, setColourMode] = useState<ColourMode>('book');
  const [hovered, setHovered] = useState<HoveredPassage | null>(null);

  useEffect(() => {
    fetch('/story_shapes.json')
      .then((r) => r.json())
      .then((d: StoryData) => setData(d));
  }, []);

  const books = data?.books ?? [];

  const singleActiveBook = useMemo(() => {
    if (activeBooks.size === 1) return [...activeBooks][0];
    return null;
  }, [activeBooks]);

  const passageCount = singleActiveBook !== null ? books[singleActiveBook]?.passages.length ?? 0 : 0;
  const playback = usePlayback(passageCount);

  useEffect(() => {
    playback.reset();
  }, [singleActiveBook]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggleBook = useCallback((index: number) => {
    setActiveBooks((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const setAllBooks = useCallback(
    (active: boolean) => {
      if (active) {
        setActiveBooks(new Set(books.map((_, i) => i)));
      } else {
        setActiveBooks(new Set());
      }
    },
    [books]
  );

  const playbackTarget = useMemo(() => {
    if (singleActiveBook === null) return null;
    return { bookIndex: singleActiveBook, passageIndex: playback.currentIndex };
  }, [singleActiveBook, playback.currentIndex]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === 'Space' && singleActiveBook !== null) {
        e.preventDefault();
        playback.togglePlay();
      } else if (e.code === 'ArrowRight' && singleActiveBook !== null) {
        e.preventDefault();
        playback.setCurrentIndex(Math.min(playback.currentIndex + 1, passageCount - 1));
      } else if (e.code === 'ArrowLeft' && singleActiveBook !== null) {
        e.preventDefault();
        playback.setCurrentIndex(Math.max(playback.currentIndex - 1, 0));
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [singleActiveBook, playback, passageCount]);

  if (!data) {
    return (
      <div className="h-screen flex items-center justify-center bg-[#f5f0e8] text-[#8a7e6b] text-xl italic">
        Loading narrative space...
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-[#f5f0e8] overflow-hidden">
      <Canvas
        books={books}
        activeBooks={activeBooks}
        colourMode={colourMode}
        hovered={hovered}
        onHover={setHovered}
        playback={playbackTarget}
      />
      <Sidebar
        books={books}
        activeBooks={activeBooks}
        colourMode={colourMode}
        onToggleBook={toggleBook}
        onSetAllBooks={setAllBooks}
        onSetColourMode={setColourMode}
        singleActiveBook={singleActiveBook}
        playbackState={singleActiveBook !== null ? playback : null}
      />
      <PassagePanel books={books} hovered={hovered} playback={playbackTarget} />
    </div>
  );
}
