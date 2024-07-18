function updateBackgroundColor() {
    const bodyClass = document.body.classList;
    const textElements = document.querySelectorAll('.dynamic-text');
  
    textElements.forEach(textElement => {
      if (bodyClass.contains('quarto-light')) {
        textElement.style.backgroundColor = '#f0f0f0'; // Light mode color
      } else if (bodyClass.contains('quarto-dark')) {
        textElement.style.backgroundColor = '#4f4f4f'; // Dark mode color
      }
    });
  }
  
  // Observe changes to the body's class attribute
  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
        updateBackgroundColor();
      }
    });
  });
  
  // Start observing
  observer.observe(document.body, {
    attributes: true
  });
  
  // Initial update
  updateBackgroundColor();
  