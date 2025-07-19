import React, { createContext, useContext, useEffect, useState } from 'react';
import { ThemeSettings } from '@/types';
import { defaultTheme } from '@/utils/theme';
import { localStorage } from '@/utils';

interface ThemeContextType {
  theme: ThemeSettings;
  setTheme: (theme: Partial<ThemeSettings>) => void;
  toggleTheme: () => void;
  isDark: boolean;
  setMode: (mode: 'light' | 'dark' | 'system') => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: React.ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setThemeState] = useState<ThemeSettings>(() => {
    const savedTheme = localStorage.get('theme', defaultTheme);
    return { ...defaultTheme, ...savedTheme };
  });

  const [isDark, setIsDark] = useState<boolean>(false);

  // Function to determine if dark mode should be active
  const shouldUseDarkMode = (mode: 'light' | 'dark' | 'system'): boolean => {
    if (mode === 'system') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return mode === 'dark';
  };

  // Update theme state and localStorage
  const setTheme = (newTheme: Partial<ThemeSettings>) => {
    const updatedTheme = { ...theme, ...newTheme };
    setThemeState(updatedTheme);
    localStorage.set('theme', updatedTheme);
  };

  // Toggle between light and dark mode
  const toggleTheme = () => {
    const newMode = isDark ? 'light' : 'dark';
    setTheme({ mode: newMode });
  };

  // Set specific mode
  const setMode = (mode: 'light' | 'dark' | 'system') => {
    setTheme({ mode });
  };

  // Effect to handle theme changes
  useEffect(() => {
    const darkModeActive = shouldUseDarkMode(theme.mode);
    setIsDark(darkModeActive);

    // Update document classes
    const root = document.documentElement;
    if (darkModeActive) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }

    // Update CSS custom properties
    root.style.setProperty('--primary-color', theme.primaryColor);
    root.style.setProperty('--secondary-color', theme.secondaryColor);
    root.style.setProperty('--accent-color', theme.accentColor);

    // Update meta theme-color
    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement('meta');
      metaThemeColor.setAttribute('name', 'theme-color');
      document.head.appendChild(metaThemeColor);
    }
    metaThemeColor.setAttribute('content', theme.primaryColor);
  }, [theme]);

  // Effect to handle system theme changes
  useEffect(() => {
    if (theme.mode === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      const handleChange = (e: MediaQueryListEvent) => {
        setIsDark(e.matches);
        
        // Update document classes
        const root = document.documentElement;
        if (e.matches) {
          root.classList.add('dark');
        } else {
          root.classList.remove('dark');
        }
      };

      mediaQuery.addEventListener('change', handleChange);
      return () => mediaQuery.removeEventListener('change', handleChange);
    }
  }, [theme.mode]);

  // Effect to handle reduced motion preference
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    
    const handleChange = (e: MediaQueryListEvent) => {
      if (e.matches && !theme.reducedMotion) {
        setTheme({ reducedMotion: true, animations: false });
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    
    // Check initial state
    if (mediaQuery.matches && !theme.reducedMotion) {
      setTheme({ reducedMotion: true, animations: false });
    }

    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme.reducedMotion, theme.animations]);

  // Effect to apply animation preferences
  useEffect(() => {
    const root = document.documentElement;
    
    if (theme.reducedMotion || !theme.animations) {
      root.style.setProperty('--animation-duration', '0s');
      root.style.setProperty('--transition-duration', '0s');
    } else {
      root.style.setProperty('--animation-duration', '0.3s');
      root.style.setProperty('--transition-duration', '0.2s');
    }
  }, [theme.animations, theme.reducedMotion]);

  const contextValue: ThemeContextType = {
    theme,
    setTheme,
    toggleTheme,
    isDark,
    setMode,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

// Hook to get theme-specific colors
export const useThemeColors = () => {
  const { theme, isDark } = useTheme();
  
  const colors = {
    primary: theme.primaryColor,
    secondary: theme.secondaryColor,
    accent: theme.accentColor,
    background: isDark ? '#0F172A' : '#FFFFFF',
    surface: isDark ? '#1E293B' : '#F8FAFC',
    text: isDark ? '#F1F5F9' : '#0F172A',
    textSecondary: isDark ? '#94A3B8' : '#64748B',
    border: isDark ? '#334155' : '#E2E8F0',
    ...theme.customColors,
  };

  return colors;
};

// Hook to get responsive theme values
export const useResponsiveTheme = () => {
  const { theme } = useTheme();
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);

  useEffect(() => {
    const checkScreenSize = () => {
      const width = window.innerWidth;
      setIsMobile(width < 768);
      setIsTablet(width >= 768 && width < 1024);
    };

    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  return {
    theme,
    isMobile,
    isTablet,
    isDesktop: !isMobile && !isTablet,
  };
};

// Custom hook for theme-aware animations
export const useThemeAnimation = () => {
  const { theme } = useTheme();
  
  const getAnimationClass = (animationName: string) => {
    if (!theme.animations || theme.reducedMotion) {
      return '';
    }
    return animationName;
  };

  const getTransitionClass = (transitionName: string) => {
    if (!theme.animations || theme.reducedMotion) {
      return '';
    }
    return transitionName;
  };

  return {
    getAnimationClass,
    getTransitionClass,
    animationsEnabled: theme.animations && !theme.reducedMotion,
  };
};

export default ThemeProvider;