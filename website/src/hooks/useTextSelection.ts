import { useState, useEffect } from 'react';

interface TextSelection {
  text: string;
  context: string;
  element: HTMLElement | null;
}

export const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState<TextSelection | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);

  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().trim() !== '') {
        const selectedTextContent = selection.toString().trim();
        if (selectedTextContent.length > 0) {
          // Get the context around the selection
          const range = selection.getRangeAt(0);
          const element = range.commonAncestorContainer.parentElement;

          // Extract some context around the selection
          const contextNode = range.cloneContents();
          const contextText = contextNode.textContent || '';

          setSelectedText({
            text: selectedTextContent,
            context: contextText.substring(0, 500), // Limit context to 500 chars
            element: element
          });
          setIsSelecting(true);
        }
      } else {
        setSelectedText(null);
        setIsSelecting(false);
      }
    };

    const handleMouseUp = () => {
      setTimeout(handleSelection, 0); // Allow selection to complete
    };

    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Escape') {
        const selection = window.getSelection();
        if (selection) selection.removeAllRanges();
        setSelectedText(null);
        setIsSelecting(false);
      }
    });

    return () => {
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
          const selection = window.getSelection();
          if (selection) selection.removeAllRanges();
        }
      });
    };
  }, []);

  const clearSelection = () => {
    const selection = window.getSelection();
    if (selection) selection.removeAllRanges();
    setSelectedText(null);
    setIsSelecting(false);
  };

  return {
    selectedText,
    isSelecting,
    clearSelection
  };
};

export default useTextSelection;