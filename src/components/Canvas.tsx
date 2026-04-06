import { useRef, useEffect, useCallback, useState } from 'react';
import type { Book, ColourMode, HoveredPassage } from '../utils/types';
import { useCanvasRenderer } from '../hooks/useCanvasRenderer';
import { useHitTest } from '../hooks/useHitTest';

interface Props {
  books: Book[];
  activeBooks: Set<number>;
  colourMode: ColourMode;
  hovered: HoveredPassage | null;
  onHover: (h: HoveredPassage | null) => void;
  playback: { bookIndex: number; passageIndex: number } | null;
}

function fitTransform(books: Book[], width: number, height: number) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const book of books) {
    for (const p of book.passages) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const padding = 60;
  const scaleX = (width - padding * 2) / rangeX;
  const scaleY = (height - padding * 2) / rangeY;
  const scale = Math.min(scaleX, scaleY);
  const offsetX = (width - rangeX * scale) / 2 - minX * scale;
  const offsetY = (height - rangeY * scale) / 2 - minY * scale;
  return { offsetX, offsetY, scale };
}

export function Canvas({ books, activeBooks, colourMode, hovered, onHover, playback }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [transform, setTransform] = useState({ offsetX: 0, offsetY: 0, scale: 1 });
  const initialFitDone = useRef(false);
  const dragRef = useRef<{ startX: number; startY: number; startOX: number; startOY: number } | null>(null);

  const { render } = useCanvasRenderer();
  const { hitTest } = useHitTest(books, transform, size.width, size.height);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Fit data on load
  useEffect(() => {
    if (books.length > 0 && size.width > 100 && !initialFitDone.current) {
      setTransform(fitTransform(books, size.width, size.height));
      initialFitDone.current = true;
    }
  }, [books, size]);

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render(ctx, size.width, size.height, {
      books,
      activeBooks,
      colourMode,
      hovered,
      playback,
      transform,
    });
  }, [books, activeBooks, colourMode, hovered, playback, transform, size, render]);

  // Mouse move → hit test
  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const hit = hitTest(mx, my);
      onHover(hit ? { ...hit, screenX: e.clientX, screenY: e.clientY } : null);
    },
    [hitTest, onHover]
  );

  const handleMouseLeave = useCallback(() => onHover(null), [onHover]);

  // Zoom
  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      setTransform((t) => ({
        scale: t.scale * factor,
        offsetX: mx - (mx - t.offsetX) * factor,
        offsetY: my - (my - t.offsetY) * factor,
      }));
    },
    []
  );

  // Pan
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startOX: transform.offsetX,
        startOY: transform.offsetY,
      };
    },
    [transform]
  );

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  const handleDragMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragRef.current) return;
      const dx = e.clientX - dragRef.current.startX;
      const dy = e.clientY - dragRef.current.startY;
      setTransform((t) => ({
        ...t,
        offsetX: dragRef.current!.startOX + dx,
        offsetY: dragRef.current!.startOY + dy,
      }));
    },
    []
  );

  // Double-click reset
  const handleDoubleClick = useCallback(() => {
    if (books.length > 0) {
      setTransform(fitTransform(books, size.width, size.height));
    }
  }, [books, size]);

  return (
    <div ref={containerRef} className="flex-1 h-screen overflow-hidden relative">
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        onMouseMove={(e) => {
          handleMouseMove(e);
          handleDragMove(e);
        }}
        onMouseLeave={handleMouseLeave}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onDoubleClick={handleDoubleClick}
      />
    </div>
  );
}
