// Referencias a elementos del DOM
const predictBtn = document.getElementById("predictBtn");
const trainBtn = document.getElementById("trainBtn");
const reviewInput = document.getElementById("reviewInput");
const resultDiv = document.getElementById("result");
const trainStatusDiv = document.getElementById("trainStatus");

// Evento para predecir la reseña
predictBtn.addEventListener("click", async () => {
  const reviewText = reviewInput.value.trim();

  if (!reviewText) {
    resultDiv.textContent = "Por favor, ingresa un texto de reseña.";
    return;
  }

  try {
    // Llamada al endpoint /predict en el backend
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: reviewText })
    });

    if (!response.ok) {
      throw new Error("Error en la respuesta del servidor.");
    }

    const data = await response.json();

    resultDiv.textContent = `La reseña se clasifica como: ${data.prediction}`;
  } catch (error) {
    console.error(error);
    resultDiv.textContent = "Ocurrió un error al intentar predecir.";
  }
});

// Evento para entrenar el modelo
trainBtn.addEventListener("click", async () => {
  try {
    const response = await fetch("http://localhost:8000/train", {
      method: "GET"
    });

    if (!response.ok) {
      throw new Error("Error en la respuesta del servidor al entrenar.");
    }

    const data = await response.json();
    trainStatusDiv.textContent = data.message || "Entrenamiento finalizado correctamente.";
  } catch (error) {
    console.error(error);
    trainStatusDiv.textContent = "Error al entrenar el modelo.";
  }
});
