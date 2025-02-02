function sendMessage() {
    const userMessage = document.getElementById("userMessage").value;
    if (!userMessage) return;

    // Ajouter le message de l'utilisateur à la boîte de chat
    const chatBody = document.querySelector(".chat-body");
    chatBody.innerHTML += `<div class="user-message">${userMessage}</div>`;

    // Envoyer le message au backend Flask
    fetch("/send_message", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage }),
    })
    .then((response) => response.json())
    .then((data) => {
        // Ajouter la réponse du chatbot à la boîte de chat
        chatBody.innerHTML += `<div class="bot-message">${data.response}</div>`;
        chatBody.scrollTop = chatBody.scrollHeight; // Faire défiler vers le bas
    })
    .catch((error) => {
        console.error("Erreur:", error);
    });

    // Effacer le champ de saisie
    document.getElementById("userMessage").value = "";
}