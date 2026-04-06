export interface Passage {
  x: number;
  y: number;
  x3d: number;
  y3d: number;
  z3d: number;
  pos: number;
  text: string;
  dialogue: number;
  entropy: number;
  sent_len: number;
}

export interface Book {
  title: string;
  full_length: number;
  archetype: number;
  passages: Passage[];
}

export interface Metadata {
  model: string;
  layer: number;
  window: number;
  ae_k: number;
  n_books: number;
  n_passages: number;
}

export interface StoryData {
  metadata: Metadata;
  books: Book[];
}

export type ColourMode = 'book' | 'dialogue' | 'entropy' | 'sent_len' | 'position';

export interface HoveredPassage {
  bookIndex: number;
  passageIndex: number;
  screenX: number;
  screenY: number;
}
