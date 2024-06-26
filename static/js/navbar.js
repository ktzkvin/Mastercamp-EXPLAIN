document.addEventListener("DOMContentLoaded", function () {
    const navbar = document.getElementById("navbar");
    navbar.classList.add("default"); // Ajouter la classe default initialement
    window.addEventListener("scroll", function () {
        if (window.scrollY > 50) {
            navbar.classList.remove("default");
            navbar.classList.add("scrolled");
        } else {
            navbar.classList.remove("scrolled");
            navbar.classList.add("default");
        }
    });
});
