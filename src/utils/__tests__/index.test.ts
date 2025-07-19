import { describe, it, expect, vi } from 'vitest'
import { 
  formatNumber, 
  formatFeatureValue, 
  debounce, 
  deepClone, 
  isEmpty,
  calculatePercentage,
  sortBy,
  groupBy,
  generateColorFromString,
  capitalizeFirst,
  camelToTitle,
  snakeToTitle
} from '../index'

describe('Utils', () => {
  describe('formatNumber', () => {
    it('formats numbers correctly', () => {
      expect(formatNumber(123.456)).toBe('123.46')
      expect(formatNumber(123.456, { precision: 1 })).toBe('123.5')
      expect(formatNumber(0.123, { percentage: true })).toBe('0.12%')
      expect(formatNumber(1000, { compact: true })).toBe('1K')
    })

    it('handles edge cases', () => {
      expect(formatNumber(NaN)).toBe('N/A')
      expect(formatNumber(Infinity)).toBe('N/A')
      expect(formatNumber(-Infinity)).toBe('N/A')
    })
  })

  describe('formatFeatureValue', () => {
    it('formats different types correctly', () => {
      expect(formatFeatureValue(123.45, 'numerical')).toBe('123.45')
      expect(formatFeatureValue('category', 'categorical')).toBe('category')
      expect(formatFeatureValue(true, 'boolean')).toBe('Yes')
      expect(formatFeatureValue(false, 'boolean')).toBe('No')
      expect(formatFeatureValue(null, 'numerical')).toBe('N/A')
    })
  })

  describe('debounce', () => {
    it('debounces function calls', async () => {
      const mockFn = vi.fn()
      const debouncedFn = debounce(mockFn, 100)
      
      debouncedFn()
      debouncedFn()
      debouncedFn()
      
      expect(mockFn).not.toHaveBeenCalled()
      
      await new Promise(resolve => setTimeout(resolve, 150))
      expect(mockFn).toHaveBeenCalledTimes(1)
    })
  })

  describe('deepClone', () => {
    it('creates deep copies of objects', () => {
      const original = { a: 1, b: { c: 2 } }
      const cloned = deepClone(original)
      
      expect(cloned).toEqual(original)
      expect(cloned).not.toBe(original)
      expect(cloned.b).not.toBe(original.b)
    })

    it('handles arrays', () => {
      const original = [1, [2, 3], { a: 4 }]
      const cloned = deepClone(original)
      
      expect(cloned).toEqual(original)
      expect(cloned).not.toBe(original)
      expect(cloned[1]).not.toBe(original[1])
    })
  })

  describe('isEmpty', () => {
    it('correctly identifies empty values', () => {
      expect(isEmpty(null)).toBe(true)
      expect(isEmpty(undefined)).toBe(true)
      expect(isEmpty('')).toBe(true)
      expect(isEmpty('  ')).toBe(true)
      expect(isEmpty([])).toBe(true)
      expect(isEmpty({})).toBe(true)
      
      expect(isEmpty('hello')).toBe(false)
      expect(isEmpty([1])).toBe(false)
      expect(isEmpty({ a: 1 })).toBe(false)
    })
  })

  describe('calculatePercentage', () => {
    it('calculates percentages correctly', () => {
      expect(calculatePercentage(50, 100)).toBe(50)
      expect(calculatePercentage(25, 200)).toBe(12.5)
      expect(calculatePercentage(10, 0)).toBe(0)
    })
  })

  describe('sortBy', () => {
    it('sorts arrays by property', () => {
      const data = [
        { name: 'Bob', age: 30 },
        { name: 'Alice', age: 25 },
        { name: 'Charlie', age: 35 }
      ]
      
      const sortedByAge = sortBy(data, 'age')
      expect(sortedByAge[0].name).toBe('Alice')
      expect(sortedByAge[2].name).toBe('Charlie')
      
      const sortedByAgeDesc = sortBy(data, 'age', 'desc')
      expect(sortedByAgeDesc[0].name).toBe('Charlie')
      expect(sortedByAgeDesc[2].name).toBe('Alice')
    })
  })

  describe('groupBy', () => {
    it('groups array by property', () => {
      const data = [
        { type: 'A', value: 1 },
        { type: 'B', value: 2 },
        { type: 'A', value: 3 }
      ]
      
      const grouped = groupBy(data, 'type')
      expect(grouped.A).toHaveLength(2)
      expect(grouped.B).toHaveLength(1)
    })
  })

  describe('generateColorFromString', () => {
    it('generates consistent colors for strings', () => {
      const color1 = generateColorFromString('test')
      const color2 = generateColorFromString('test')
      expect(color1).toBe(color2)
      
      const color3 = generateColorFromString('different')
      expect(color1).not.toBe(color3)
    })
  })

  describe('string formatting', () => {
    it('capitalizes first letter', () => {
      expect(capitalizeFirst('hello')).toBe('Hello')
      expect(capitalizeFirst('HELLO')).toBe('HELLO')
      expect(capitalizeFirst('')).toBe('')
    })

    it('converts camelCase to Title Case', () => {
      expect(camelToTitle('camelCase')).toBe('Camel Case')
      expect(camelToTitle('someVariableName')).toBe('Some Variable Name')
    })

    it('converts snake_case to Title Case', () => {
      expect(snakeToTitle('snake_case')).toBe('Snake Case')
      expect(snakeToTitle('some_variable_name')).toBe('Some Variable Name')
    })
  })
})