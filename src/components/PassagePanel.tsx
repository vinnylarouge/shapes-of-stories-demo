import type { Book, HoveredPassage } from '../utils/types';
import { BOOK_COLOURS } from '../utils/colours';

interface Props {
  books: Book[];
  hovered: HoveredPassage | null;
  playback: { bookIndex: number; passageIndex: number } | null;
}

export function PassagePanel({ books, hovered, playback }: Props) {
  const target = hovered ?? playback;
  if (!target) {
    return (
      <div className="absolute bottom-4 left-4 right-[316px] bg-[#ece5d8]/90 backdrop-blur border border-[#d4cbb8] rounded px-5 py-3 text-[#b5a992] text-sm italic pointer-events-none">
        Hover over a point to read its passage...
      </div>
    );
  }

  const book = books[target.bookIndex];
  const passage = book.passages[target.passageIndex];
  const colour = BOOK_COLOURS[target.bookIndex % BOOK_COLOURS.length];
  const pct = Math.round(passage.pos * 100);

  return (
    <div className="absolute bottom-4 left-4 right-[316px] bg-[#ece5d8]/90 backdrop-blur border border-[#d4cbb8] rounded px-5 py-3 pointer-events-none">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: colour }} />
        <span className="text-[#3d3328] text-sm font-medium">{book.title}</span>
        <span className="text-[#8a7e6b] text-xs ml-auto">{pct}% through</span>
      </div>
      <p className="text-[#5a4d3e] text-sm leading-relaxed mb-2 italic">"{passage.text}"</p>
      <div className="flex gap-5 text-xs text-[#8a7e6b]">
        <span>Dialogue: {(passage.dialogue * 100).toFixed(0)}%</span>
        <span>Entropy: {passage.entropy.toFixed(2)}</span>
        <span>Avg. sentence: {passage.sent_len.toFixed(1)} words</span>
      </div>
    </div>
  );
}
