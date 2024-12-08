// Handle form submission and dynamically display results
document.querySelector("form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(event.target);
    const response = await fetch("/search", {
        method: "POST",
        body: formData,
    });

    const resultsDiv = document.querySelector(".results");
    resultsDiv.innerHTML = "<h2>Top Search Results</h2>";

    if (response.ok) {
        const results = await response.json();
        results.forEach((result) => {
            const resultItem = document.createElement("div");
            resultItem.classList.add("result-item");

            const img = document.createElement("img");
            img.src = result.file_name;
            img.alt = "Result Image";

            const similarity = document.createElement("p");
            similarity.textContent = `Similarity: ${result.score.toFixed(3)}`;

            resultItem.appendChild(img);
            resultItem.appendChild(similarity);
            resultsDiv.appendChild(resultItem);
        });
    } else {
        resultsDiv.innerHTML = "<p>Error fetching results. Please try again.</p>";
    }
});
