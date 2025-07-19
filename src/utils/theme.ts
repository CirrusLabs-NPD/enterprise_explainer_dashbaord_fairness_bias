import { ThemeSettings } from '@/types';

export const defaultTheme: ThemeSettings = {
  mode: 'light',
  primaryColor: '#3B82F6',
  secondaryColor: '#10B981',
  accentColor: '#7C3AED',
  animations: true,
  reducedMotion: false,
};

export const darkTheme: ThemeSettings = {
  mode: 'dark',
  primaryColor: '#60A5FA',
  secondaryColor: '#34D399',
  accentColor: '#A78BFA',
  animations: true,
  reducedMotion: false,
};

export const colorPalettes = {
  primary: {
    50: '#EFF6FF',
    100: '#DBEAFE',
    200: '#BFDBFE',
    300: '#93C5FD',
    400: '#60A5FA',
    500: '#3B82F6',
    600: '#2563EB',
    700: '#1D4ED8',
    800: '#1E40AF',
    900: '#1E3A8A',
  },
  secondary: {
    50: '#ECFDF5',
    100: '#D1FAE5',
    200: '#A7F3D0',
    300: '#6EE7B7',
    400: '#34D399',
    500: '#10B981',
    600: '#059669',
    700: '#047857',
    800: '#065F46',
    900: '#064E3B',
  },
  accent: {
    50: '#FAF5FF',
    100: '#F3E8FF',
    200: '#E9D5FF',
    300: '#D8B4FE',
    400: '#C084FC',
    500: '#A855F7',
    600: '#9333EA',
    700: '#7C3AED',
    800: '#6B21A8',
    900: '#581C87',
  },
  neutral: {
    50: '#F8FAFC',
    100: '#F1F5F9',
    200: '#E2E8F0',
    300: '#CBD5E1',
    400: '#94A3B8',
    500: '#64748B',
    600: '#475569',
    700: '#334155',
    800: '#1E293B',
    900: '#0F172A',
  },
  success: {
    50: '#F0FDF4',
    100: '#DCFCE7',
    200: '#BBF7D0',
    300: '#86EFAC',
    400: '#4ADE80',
    500: '#22C55E',
    600: '#16A34A',
    700: '#15803D',
    800: '#166534',
    900: '#14532D',
  },
  warning: {
    50: '#FFFBEB',
    100: '#FEF3C7',
    200: '#FDE68A',
    300: '#FCD34D',
    400: '#FBBF24',
    500: '#F59E0B',
    600: '#D97706',
    700: '#B45309',
    800: '#92400E',
    900: '#78350F',
  },
  error: {
    50: '#FEF2F2',
    100: '#FEE2E2',
    200: '#FECACA',
    300: '#FCA5A5',
    400: '#F87171',
    500: '#EF4444',
    600: '#DC2626',
    700: '#B91C1C',
    800: '#991B1B',
    900: '#7F1D1D',
  },
};

export const chartColors = {
  qualitative: [
    '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', 
    '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#14B8A6'
  ],
  sequential: [
    '#EFF6FF', '#DBEAFE', '#BFDBFE', '#93C5FD', '#60A5FA',
    '#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF', '#1E3A8A'
  ],
  diverging: [
    '#DC2626', '#EF4444', '#F87171', '#FCA5A5', '#FECACA',
    '#E5E7EB', '#C3C7D4', '#9CA3AF', '#6B7280', '#374151'
  ],
  gradient: {
    primary: 'linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%)',
    secondary: 'linear-gradient(135deg, #059669 0%, #10B981 100%)',
    accent: 'linear-gradient(135deg, #7C3AED 0%, #EC4899 100%)',
    success: 'linear-gradient(135deg, #16A34A 0%, #22C55E 100%)',
    warning: 'linear-gradient(135deg, #D97706 0%, #F59E0B 100%)',
    error: 'linear-gradient(135deg, #DC2626 0%, #EF4444 100%)',
  },
};

export const animations = {
  duration: {
    fast: 150,
    normal: 300,
    slow: 500,
  },
  easing: {
    linear: 'linear',
    easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
    easeOut: 'cubic-bezier(0, 0, 0.2, 1)',
    easeInOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
};

export const breakpoints = {
  xs: '480px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
};

export const spacing = {
  0: '0',
  1: '0.25rem',
  2: '0.5rem',
  3: '0.75rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  8: '2rem',
  10: '2.5rem',
  12: '3rem',
  16: '4rem',
  20: '5rem',
  24: '6rem',
  32: '8rem',
  40: '10rem',
  48: '12rem',
  56: '14rem',
  64: '16rem',
};

export const typography = {
  fontFamily: {
    sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
    mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
  },
  fontSize: {
    xs: '0.75rem',
    sm: '0.875rem',
    base: '1rem',
    lg: '1.125rem',
    xl: '1.25rem',
    '2xl': '1.5rem',
    '3xl': '1.875rem',
    '4xl': '2.25rem',
    '5xl': '3rem',
    '6xl': '3.75rem',
  },
  fontWeight: {
    light: 300,
    normal: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
    extrabold: 800,
  },
  lineHeight: {
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 2,
  },
};

export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  glass: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
  'glass-dark': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
};

export const borderRadius = {
  none: '0',
  sm: '0.125rem',
  base: '0.25rem',
  md: '0.375rem',
  lg: '0.5rem',
  xl: '0.75rem',
  '2xl': '1rem',
  '3xl': '1.5rem',
  full: '9999px',
};

export const zIndex = {
  auto: 'auto',
  0: '0',
  10: '10',
  20: '20',
  30: '30',
  40: '40',
  50: '50',
  dropdown: '1000',
  sticky: '1020',
  fixed: '1030',
  modalBackdrop: '1040',
  modal: '1050',
  popover: '1060',
  tooltip: '1070',
  toast: '1080',
};

export function getThemeColor(color: string, shade: number = 500): string {
  const [colorName, colorShade] = color.split('-');
  const palette = colorPalettes[colorName as keyof typeof colorPalettes];
  
  if (!palette) {
    console.warn(`Color palette '${colorName}' not found`);
    return colorPalettes.neutral[500];
  }
  
  const targetShade = colorShade ? parseInt(colorShade) : shade;
  const colorValue = palette[targetShade as keyof typeof palette];
  
  if (!colorValue) {
    console.warn(`Color shade '${targetShade}' not found in palette '${colorName}'`);
    return palette[500];
  }
  
  return colorValue;
}

export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

export function rgbToHex(r: number, g: number, b: number): string {
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

export function getContrastColor(hex: string): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return '#000000';
  
  const { r, g, b } = rgb;
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  
  return brightness > 128 ? '#000000' : '#FFFFFF';
}

export function lightenColor(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  
  const { r, g, b } = rgb;
  const factor = 1 + (percent / 100);
  
  return rgbToHex(
    Math.min(255, Math.round(r * factor)),
    Math.min(255, Math.round(g * factor)),
    Math.min(255, Math.round(b * factor))
  );
}

export function darkenColor(hex: string, percent: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  
  const { r, g, b } = rgb;
  const factor = 1 - (percent / 100);
  
  return rgbToHex(
    Math.max(0, Math.round(r * factor)),
    Math.max(0, Math.round(g * factor)),
    Math.max(0, Math.round(b * factor))
  );
}

export function addAlpha(hex: string, alpha: number): string {
  const rgb = hexToRgb(hex);
  if (!rgb) return hex;
  
  const { r, g, b } = rgb;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function generateGradient(startColor: string, endColor: string, direction: string = '135deg'): string {
  return `linear-gradient(${direction}, ${startColor} 0%, ${endColor} 100%)`;
}

export function getRandomColor(): string {
  return chartColors.qualitative[Math.floor(Math.random() * chartColors.qualitative.length)];
}

export function getRandomColors(count: number): string[] {
  const colors = [];
  for (let i = 0; i < count; i++) {
    colors.push(chartColors.qualitative[i % chartColors.qualitative.length]);
  }
  return colors;
}

export function interpolateColor(color1: string, color2: string, factor: number): string {
  const rgb1 = hexToRgb(color1);
  const rgb2 = hexToRgb(color2);
  
  if (!rgb1 || !rgb2) return color1;
  
  const r = Math.round(rgb1.r + (rgb2.r - rgb1.r) * factor);
  const g = Math.round(rgb1.g + (rgb2.g - rgb1.g) * factor);
  const b = Math.round(rgb1.b + (rgb2.b - rgb1.b) * factor);
  
  return rgbToHex(r, g, b);
}

export function getColorScale(colors: string[], steps: number): string[] {
  if (colors.length === 0) return [];
  if (colors.length === 1) return Array(steps).fill(colors[0]);
  if (steps <= colors.length) return colors.slice(0, steps);
  
  const scale = [];
  const segmentSize = (steps - 1) / (colors.length - 1);
  
  for (let i = 0; i < steps; i++) {
    const segmentIndex = i / segmentSize;
    const lowerIndex = Math.floor(segmentIndex);
    const upperIndex = Math.min(lowerIndex + 1, colors.length - 1);
    const factor = segmentIndex - lowerIndex;
    
    scale.push(interpolateColor(colors[lowerIndex], colors[upperIndex], factor));
  }
  
  return scale;
}

export default {
  colorPalettes,
  chartColors,
  animations,
  breakpoints,
  spacing,
  typography,
  shadows,
  borderRadius,
  zIndex,
  getThemeColor,
  hexToRgb,
  rgbToHex,
  getContrastColor,
  lightenColor,
  darkenColor,
  addAlpha,
  generateGradient,
  getRandomColor,
  getRandomColors,
  interpolateColor,
  getColorScale,
};