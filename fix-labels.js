// Fix Gradio label colors to black
function fixLabels() {
    document.querySelectorAll('span[data-testid="block-info"]').forEach(el => {
        el.style.color = '#000000';
        el.style.fontWeight = '700';
    });
}

// Run on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', fixLabels);
} else {
    fixLabels();
}

// Run with delays for dynamic elements
setTimeout(fixLabels, 50);
setTimeout(fixLabels, 200);
setTimeout(fixLabels, 500);

// Observe for any new labels that appear dynamically
const observer = new MutationObserver(fixLabels);
observer.observe(document.documentElement, { 
    childList: true, 
    subtree: true
});
