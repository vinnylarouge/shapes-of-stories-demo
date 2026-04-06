import { useCallback } from 'react';
import type { Book, ColourMode } from '../utils/types';
import { BOOK_COLOURS, getFeatureColour } from '../utils/colours';

interface RenderOptions {
  books: Book[];
  activeBooks: Set<number>;
  colourMode: ColourMode;
  hovered: { bookIndex: number; passageIndex: number } | null;
  playback: { bookIndex: number; passageIndex: number } | null;
  transform: { offsetX: number; offsetY: number; scale: number };
}

export function useCanvasRenderer() {
  const render = useCallback(
    (ctx: CanvasRenderingContext2D, width: number, height: number, opts: RenderOptions) => {
      const { books, activeBooks, colourMode, hovered, playback, transform } = opts;
      const { offsetX, offsetY, scale } = transform;

      const toScreen = (x: number, y: number) => ({
        sx: x * scale + offsetX,
        sy: y * scale + offsetY,
      });

      ctx.clearRect(0, 0, width, height);
      // Warm ivory/parchment background
      ctx.fillStyle = '#f5f0e8';
      ctx.fillRect(0, 0, width, height);

      // Pass 1: inactive dots — tinted by book colour so clusters are visible
      for (let bi = 0; bi < books.length; bi++) {
        if (activeBooks.has(bi)) continue;
        const passages = books[bi].passages;
        const bookColour = BOOK_COLOURS[bi % BOOK_COLOURS.length];
        for (let pi = 0; pi < passages.length; pi++) {
          const p = passages[pi];
          const { sx, sy } = toScreen(p.x, p.y);
          if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue;

          let colour: string;
          if (colourMode === 'book') {
            colour = bookColour + '40';
          } else {
            const fc = getFeatureColour(colourMode, p);
            colour = fc + '77';
          }
          ctx.beginPath();
          ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = colour;
          ctx.fill();
        }
      }

      // Pass 2: active book trajectories
      for (const bi of activeBooks) {
        const book = books[bi];
        const passages = book.passages;
        const bookColour = BOOK_COLOURS[bi % BOOK_COLOURS.length];

        // Lines — thin, elegant
        ctx.beginPath();
        ctx.strokeStyle = bookColour + '50';
        ctx.lineWidth = 1.2;
        for (let pi = 0; pi < passages.length; pi++) {
          const { sx, sy } = toScreen(passages[pi].x, passages[pi].y);
          if (pi === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();

        // Dots
        for (let pi = 0; pi < passages.length; pi++) {
          const p = passages[pi];
          const { sx, sy } = toScreen(p.x, p.y);
          if (sx < -10 || sx > width + 10 || sy < -10 || sy > height + 10) continue;

          let dotColour: string;
          if (colourMode === 'book') {
            dotColour = bookColour;
          } else {
            dotColour = getFeatureColour(colourMode, p);
          }

          ctx.beginPath();
          ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = dotColour;
          ctx.fill();
        }

        // Start marker (circle with dark border)
        if (passages.length > 0) {
          const start = toScreen(passages[0].x, passages[0].y);
          ctx.beginPath();
          ctx.arc(start.sx, start.sy, 6, 0, Math.PI * 2);
          ctx.fillStyle = bookColour;
          ctx.fill();
          ctx.strokeStyle = '#3d3328';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        // End marker (square with dark border)
        if (passages.length > 1) {
          const end = toScreen(passages[passages.length - 1].x, passages[passages.length - 1].y);
          ctx.fillStyle = bookColour;
          ctx.fillRect(end.sx - 5, end.sy - 5, 10, 10);
          ctx.strokeStyle = '#3d3328';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(end.sx - 5, end.sy - 5, 10, 10);
        }
      }

      // Pass 3: playback trail + highlight
      if (playback) {
        const book = books[playback.bookIndex];
        const bookColour = BOOK_COLOURS[playback.bookIndex % BOOK_COLOURS.length];
        const trailStart = Math.max(0, playback.passageIndex - 10);

        for (let pi = trailStart; pi <= playback.passageIndex; pi++) {
          const p = book.passages[pi];
          const { sx, sy } = toScreen(p.x, p.y);
          const age = playback.passageIndex - pi;
          const alpha = 1 - age / 11;
          const radius = 3.5 + (1 - age / 11) * 4;

          ctx.beginPath();
          ctx.arc(sx, sy, radius, 0, Math.PI * 2);
          ctx.fillStyle = bookColour;
          ctx.globalAlpha = alpha;
          ctx.fill();
          ctx.globalAlpha = 1;
        }

        // Ring on current
        const curr = book.passages[playback.passageIndex];
        const { sx, sy } = toScreen(curr.x, curr.y);
        ctx.beginPath();
        ctx.arc(sx, sy, 12, 0, Math.PI * 2);
        ctx.strokeStyle = bookColour;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.4;
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.beginPath();
        ctx.arc(sx, sy, 7, 0, Math.PI * 2);
        ctx.fillStyle = bookColour;
        ctx.fill();
        ctx.strokeStyle = '#3d3328';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Pass 4: hover highlight
      if (hovered) {
        const p = books[hovered.bookIndex].passages[hovered.passageIndex];
        const { sx, sy } = toScreen(p.x, p.y);
        const bookColour = BOOK_COLOURS[hovered.bookIndex % BOOK_COLOURS.length];

        // Highlight other dots from same book
        const samebook = books[hovered.bookIndex].passages;
        for (let pi = 0; pi < samebook.length; pi++) {
          if (pi === hovered.passageIndex) continue;
          const sp = toScreen(samebook[pi].x, samebook[pi].y);
          ctx.beginPath();
          ctx.arc(sp.sx, sp.sy, 3, 0, Math.PI * 2);
          ctx.fillStyle = bookColour + '30';
          ctx.fill();
        }

        // Ring on hovered dot
        ctx.beginPath();
        ctx.arc(sx, sy, 10, 0, Math.PI * 2);
        ctx.strokeStyle = '#3d3328';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = bookColour;
        ctx.fill();
      }
    },
    []
  );

  return { render };
}
