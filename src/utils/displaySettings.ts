// Centralised metadata for every display setting in the visualisation.
//
// Each entry has a short button label, a longer informative name, and a
// 1–3-sentence blurb explaining what the setting reveals and how it
// relates to *concept extraction* — the project's core question of which
// structures the autoencoded latent has learned to carve out of the
// transformer's hidden states.
//
// The Sidebar reads from this table for both the button text and the
// inline explanation block, so all naming is updated in one place.

import type { ColourMode, ViewMode } from './types';

export interface DisplaySetting {
  label: string;   // short button text
  name: string;    // longer informative name (used in the blurb header)
  blurb: string;   // 1–3 sentences
}

export const VIEW_MODE_INFO: Record<ViewMode, DisplaySetting> = {
  default: {
    label: 'Latent',
    name: 'UMAP latent — the raw concept space',
    blurb:
      "3D UMAP projection of the autoencoded transformer hidden states. Each point " +
      "is a passage; nearby points are passages the model represents similarly.",
  },
  canonical: {
    label: 'Canonical',
    name: 'Canonical axes — cross-book CCA',
    blurb:
      "Re-projects passages onto axes that vary similarly across different books. " +
      "Suppresses book-specific content; what's left is the structure recurring " +
      "across narratives — a content-independent slice of the concept space.",
  },
  interpolate: {
    label: 'Interpolate',
    name: 'Book interpolation — concept blending',
    blurb:
      "Pick two books and slide between them. Shows what concepts a hypothetical " +
      "blended narrative would traverse. Field influence bends the path along the " +
      "corpus's natural drift.",
  },
};

export const FIELD_OVERLAY_INFO: DisplaySetting = {
  label: 'Show concept drift field',
  name: 'Concept drift field — narrative gravity',
  blurb:
    "Renders the kernel-fit drift field v(z) as small arrows. Each arrow shows " +
    "where concepts tend to flow next at that point in latent space. Adaptive " +
    "octree puts more arrows where the field varies most.",
};

export const COLOUR_MODE_INFO: Record<ColourMode, DisplaySetting> = {
  book: {
    label: 'Book',
    name: 'Book identity',
    blurb:
      "Each book gets its own colour. Useful for telling trajectories apart and " +
      "seeing whether nearby passages tend to come from the same source.",
  },
  dialogue: {
    label: 'Dialogue',
    name: 'Dialogue density',
    blurb:
      "Fraction of each passage in quoted dialogue. Tests whether the latent has " +
      "carved out a 'dialogue' concept separate from narration.",
  },
  entropy: {
    label: 'Entropy',
    name: 'Token entropy',
    blurb:
      "Mean per-token prediction entropy: how surprised the language model was " +
      "by the next token. One of the simplest concepts the model itself reports.",
  },
  sent_len: {
    label: 'Sent. len',
    name: 'Mean sentence length',
    blurb:
      "Average sentence length per passage. Reveals whether the latent separates " +
      "terse from sprawling prose at the syntactic level.",
  },
  position: {
    label: 'Position',
    name: 'Position in book',
    blurb:
      "Where in its book each passage sits (0% = opening, 100% = ending). Tests " +
      "whether 'beginning of story' is itself a concept the model encodes.",
  },
  motif: {
    label: 'Motif',
    name: 'Motif code — k-means dictionary (E03)',
    blurb:
      "Each passage's nearest k-means code in the latent. Treats narrative as a " +
      "sequence over a small alphabet of recurring motifs — the simplest concept " +
      "dictionary extractable from the corpus.",
  },
  surprise: {
    label: 'Surprise',
    name: 'Drift-field surprise (E04)',
    blurb:
      "How far each passage's next step departs from the kernel-fit drift field. " +
      "Spikes mark moments the field didn't predict — chapter breaks, reveals, " +
      "scene shifts. The complement of the drift overlay.",
  },
  shape: {
    label: 'Shape',
    name: 'Topological shape archetype (E02)',
    blurb:
      "Books grouped by the H₀ persistence signature of their trajectory. Each " +
      "archetype is a topological narrative shape — linear, recurrent, branching " +
      "— extracted from geometry alone, independent of content.",
  },
  spec1: {
    label: 'Diffusion 1',
    name: 'Diffusion eigenmode 1 (E10)',
    blurb:
      "First non-trivial eigenvector of the graph Laplacian on the passage cloud. " +
      "A smooth scalar function reflecting the largest-scale axis of intrinsic " +
      "geometry — concept extraction with no text features, just latent positions.",
  },
  spec2: {
    label: 'Diffusion 2',
    name: 'Diffusion eigenmode 2 (E10)',
    blurb:
      "Second non-trivial Laplacian eigenvector. Orthogonal to mode 1 — picks up " +
      "the next axis of concept partitioning in the corpus geometry.",
  },
  spec3: {
    label: 'Diffusion 3',
    name: 'Diffusion eigenmode 3 (E10)',
    blurb:
      "Third non-trivial Laplacian eigenvector. Higher modes capture progressively " +
      "more localised structure in the corpus's intrinsic geometry.",
  },
  spec4: {
    label: 'Diffusion 4',
    name: 'Diffusion eigenmode 4 (E10)',
    blurb:
      "Fourth non-trivial Laplacian eigenvector. The bottom of the spectrum we " +
      "currently store — finer modes resolve smaller concept clusters at the cost " +
      "of more graph noise.",
  },
};
