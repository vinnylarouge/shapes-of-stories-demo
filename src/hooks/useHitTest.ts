import { useCallback } from 'react';
import type { Book } from '../utils/types';

export interface HitResult {
  bookIndex: number;
  passageIndex: number;
  screenX: number;
  screenY: number;
}

export function useHitTest(
  books: Book[],
  transform: { offsetX: number; offsetY: number; scale: number },
  canvasWidth: number,
  canvasHeight: number
) {
  const toScreen = useCallback(
    (x: number, y: number) => {
      return {
        sx: (x * transform.scale + transform.offsetX),
        sy: (y * transform.scale + transform.offsetY),
      };
    },
    [transform]
  );

  const hitTest = useCallback(
    (mouseX: number, mouseY: number): HitResult | null => {
      let closest: HitResult | null = null;
      let minDist = 12; // max pixel distance for hit

      for (let bi = 0; bi < books.length; bi++) {
        const passages = books[bi].passages;
        for (let pi = 0; pi < passages.length; pi++) {
          const { sx, sy } = toScreen(passages[pi].x, passages[pi].y);
          const dx = sx - mouseX;
          const dy = sy - mouseY;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < minDist) {
            minDist = dist;
            closest = { bookIndex: bi, passageIndex: pi, screenX: sx, screenY: sy };
          }
        }
      }
      return closest;
    },
    [books, toScreen]
  );

  return { hitTest, toScreen };
}
