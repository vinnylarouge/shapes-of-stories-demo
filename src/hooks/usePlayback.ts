import { useState, useRef, useCallback, useEffect } from 'react';

export function usePlayback(passageCount: number) {
  const [playing, setPlaying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [speed, setSpeed] = useState(1);
  const timerRef = useRef<number | null>(null);

  const stop = useCallback(() => {
    setPlaying(false);
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stop();
    setCurrentIndex(0);
  }, [stop]);

  useEffect(() => {
    if (!playing) return;
    if (timerRef.current !== null) clearInterval(timerRef.current);

    const interval = 200 / speed;
    timerRef.current = window.setInterval(() => {
      setCurrentIndex((prev) => {
        if (prev >= passageCount - 1) {
          stop();
          return prev;
        }
        return prev + 1;
      });
    }, interval);

    return () => {
      if (timerRef.current !== null) clearInterval(timerRef.current);
    };
  }, [playing, speed, passageCount, stop]);

  const togglePlay = useCallback(() => {
    if (playing) {
      stop();
    } else {
      if (currentIndex >= passageCount - 1) setCurrentIndex(0);
      setPlaying(true);
    }
  }, [playing, currentIndex, passageCount, stop]);

  return {
    playing,
    currentIndex,
    setCurrentIndex,
    speed,
    setSpeed,
    togglePlay,
    reset,
    stop,
  };
}
