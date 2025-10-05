// Architecture page interactive functionality
document.addEventListener('DOMContentLoaded', function() {
    // Find all reasoning sections that have the triangle (▶) indicator
    const reasoningSections = document.querySelectorAll('.stage-reasoning, .model-reasoning, .spec-reasoning');
    
    reasoningSections.forEach(section => {
        const header = section.querySelector('h4');
        const content = section.querySelector('ul, p');
        
        if (header && content) {
            // Add class to content for CSS targeting
            content.classList.add('reasoning-content', 'hidden');
            
            // Add click functionality
            header.addEventListener('click', function(e) {
                e.preventDefault();
                toggleReasoning(header, content, section);
            });
        }
    });
    
    // Add a "Show All" / "Hide All" button for convenience
    addToggleAllButton();
});

function toggleReasoning(header, content, section) {
    const isHidden = content.classList.contains('hidden');
    
    if (isHidden) {
        // Show content
        content.classList.remove('hidden');
        content.classList.add('visible');
        header.classList.add('expanded');
        
        // Add spacing class to parent section for proper gaps
        section.classList.add('expanded-section');
        
        // Smooth scroll if needed
        setTimeout(() => {
            smoothScrollToSection(content);
        }, 150);
    } else {
        // Hide content
        content.classList.remove('visible');
        content.classList.add('hidden');
        header.classList.remove('expanded');
        
        // Remove spacing class
        section.classList.remove('expanded-section');
    }
}

function addToggleAllButton() {
    // Create toggle all button
    const toggleAllBtn = document.createElement('button');
    toggleAllBtn.textContent = 'Show All Details';
    toggleAllBtn.className = 'toggle-all-btn';
    toggleAllBtn.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: var(--color-primary);
        color: var(--color-black);
        border: none;
        padding: 12px 20px;
        border-radius: 25px;
        font-family: var(--font-primary);
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(226, 193, 229, 0.3);
        transition: all 0.3s ease;
        z-index: 1000;
        backdrop-filter: blur(10px);
    `;
    
    let allVisible = false;
    
    toggleAllBtn.addEventListener('click', function() {
        const reasoningSections = document.querySelectorAll('.stage-reasoning, .model-reasoning, .spec-reasoning');
        
        reasoningSections.forEach(section => {
            const header = section.querySelector('h4');
            const content = section.querySelector('.reasoning-content');
            
            if (header && content) {
                if (!allVisible) {
                    // Show all
                    content.classList.remove('hidden');
                    content.classList.add('visible');
                    header.classList.add('expanded');
                } else {
                    // Hide all
                    content.classList.remove('visible');
                    content.classList.add('hidden');
                    header.classList.remove('expanded');
                }
            }
        });
        
        allVisible = !allVisible;
        toggleAllBtn.textContent = allVisible ? 'Hide All Details' : 'Show All Details';
        
        // Update button color
        if (allVisible) {
            toggleAllBtn.style.background = 'var(--color-secondary)';
        } else {
            toggleAllBtn.style.background = 'var(--color-primary)';
        }
    });
    
    // Add hover effects to the button
    toggleAllBtn.addEventListener('mouseenter', function() {
        toggleAllBtn.style.transform = 'translateY(-2px) scale(1.05)';
        toggleAllBtn.style.boxShadow = '0 6px 25px rgba(226, 193, 229, 0.4)';
    });
    
    toggleAllBtn.addEventListener('mouseleave', function() {
        toggleAllBtn.style.transform = 'translateY(0) scale(1)';
        toggleAllBtn.style.boxShadow = '0 4px 20px rgba(226, 193, 229, 0.3)';
    });
    
    document.body.appendChild(toggleAllBtn);
}

// Add smooth scrolling for better UX when sections expand
function smoothScrollToSection(element) {
    const rect = element.getBoundingClientRect();
    const isVisible = rect.top >= 0 && rect.bottom <= window.innerHeight;
    
    if (!isVisible) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'nearest'
        });
    }
}

// Add keyboard support for accessibility
document.addEventListener('keydown', function(e) {
    if (e.target.matches('.stage-reasoning h4, .model-reasoning h4, .spec-reasoning h4')) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            e.target.click();
        }
    }
});

// Add ARIA attributes for accessibility
document.addEventListener('DOMContentLoaded', function() {
    const headers = document.querySelectorAll('.stage-reasoning h4, .model-reasoning h4, .spec-reasoning h4');
    
    headers.forEach((header, index) => {
        const content = header.parentElement.querySelector('.reasoning-content');
        
        if (content) {
            header.setAttribute('role', 'button');
            header.setAttribute('aria-expanded', 'false');
            header.setAttribute('aria-controls', `reasoning-content-${index}`);
            header.setAttribute('tabindex', '0');
            
            content.setAttribute('id', `reasoning-content-${index}`);
            content.setAttribute('aria-hidden', 'true');
            
            // Update ARIA attributes when toggled
            header.addEventListener('click', function() {
                const isExpanded = header.classList.contains('expanded');
                header.setAttribute('aria-expanded', isExpanded);
                content.setAttribute('aria-hidden', !isExpanded);
            });
        }
    });
});