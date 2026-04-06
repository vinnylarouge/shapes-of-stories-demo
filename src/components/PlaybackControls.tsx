interface Props {
  playing: boolean;
  currentIndex: number;
  total: number;
  speed: number;
  onTogglePlay: () => void;
  onSetIndex: (i: number) => void;
  onSetSpeed: (s: number) => void;
  onReset: () => void;
}

export function PlaybackControls({
  playing,
  currentIndex,
  total,
  speed,
  onTogglePlay,
  onSetIndex,
  onSetSpeed,
  onReset,
}: Props) {
  return (
    <div className="px-4 py-3 border-t border-[#d4cbb8]">
      <div className="text-xs text-[#8a7e6b] mb-2 uppercase tracking-widest" style={{ fontSize: '0.65rem' }}>Playback</div>
      <div className="flex items-center gap-2 mb-2">
        <button
          onClick={onTogglePlay}
          className="px-3 py-1 text-sm rounded-sm bg-[#3d3328] hover:bg-[#5a4d3e] text-[#f5f0e8] transition-colors"
        >
          {playing ? '||' : '\u25B6'}
        </button>
        <button
          onClick={onReset}
          className="px-2 py-1 text-sm rounded-sm bg-[#ddd5c5] hover:bg-[#d4cbb8] text-[#6b5c4d] transition-colors"
        >
          \u25A0
        </button>
        <input
          type="range"
          min={0}
          max={total - 1}
          value={currentIndex}
          onChange={(e) => onSetIndex(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-xs text-[#8a7e6b] w-14 text-right tabular-nums">
          {currentIndex + 1}/{total}
        </span>
      </div>
      <div className="flex items-center gap-1">
        <span className="text-xs text-[#8a7e6b] mr-1">Speed:</span>
        {[0.5, 1, 2].map((s) => (
          <button
            key={s}
            onClick={() => onSetSpeed(s)}
            className={`px-2 py-0.5 text-xs rounded-sm transition-colors ${
              speed === s
                ? 'bg-[#3d3328] text-[#f5f0e8]'
                : 'bg-[#ddd5c5] text-[#6b5c4d] hover:bg-[#d4cbb8]'
            }`}
          >
            {s}x
          </button>
        ))}
      </div>
    </div>
  );
}
