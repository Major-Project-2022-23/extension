document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('inputForm');
    const input = document.getElementById('textInput');
    const resultContainer = document.getElementById('resultContainer');
    const resultText = document.getElementById('resultText');
  
    form.addEventListener('submit', function (event) {
      event.preventDefault(); // Prevent form submission
  
      const userInput = input.value.trim();
  
      // Perform any necessary processing on the user input
      const result = userInput ? `Hello, ${userInput}!` : 'Please enter your name.';
  
      // Display the result
      resultText.textContent = result;
      resultContainer.style.display = 'block';
  
      // Reset the form
      form.reset();
    });
  });
  