(() => {
  function ready() {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const body = document.body;

    if (prefersReducedMotion.matches) {
      body.classList.add('is-ready');
    } else {
      window.requestAnimationFrame(function () {
        body.classList.add('is-ready');
      });
    }

    initStars(prefersReducedMotion);
    initHeroCTA(prefersReducedMotion);
    initParallax(prefersReducedMotion);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready, { once: true });
  } else {
    ready();
  }

  function initStars(prefersReducedMotion) {
    if (prefersReducedMotion.matches) {
      return;
    }

    const hero = document.querySelector('.hero');
    if (!hero) {
      return;
    }

    const starCount = window.matchMedia('(min-width: 768px)').matches ? 26 : 18;
    const fragment = document.createDocumentFragment();

    for (let index = 0; index < starCount; index += 1) {
      const star = document.createElement('span');
      star.className = 'hero-star';

      const left = (Math.random() * 100).toFixed(2);
      const top = (Math.random() * 100).toFixed(2);
      const delay = (Math.random() * 6).toFixed(2);
      const duration = (4 + Math.random() * 4).toFixed(2);
      const scale = (0.6 + Math.random() * 0.8).toFixed(2);

      star.style.left = left + '%';
      star.style.top = top + '%';
      star.style.animationDelay = delay + 's';
      star.style.animationDuration = duration + 's';
      star.style.transform = 'scale(' + scale + ')';

      fragment.appendChild(star);
    }

    hero.appendChild(fragment);
  }

  function initHeroCTA(prefersReducedMotion) {
    const cta = document.querySelector('.hero__cta');
    const overlay = document.querySelector('.loading-overlay');

    if (!cta || !overlay) {
      return;
    }

    overlay.setAttribute('aria-hidden', 'true');
    let isLoading = false;

    cta.addEventListener('click', function (event) {
      if (event.defaultPrevented) {
        return;
      }

      if ((event.button && event.button !== 0) || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }

      const href = cta.getAttribute('href');
      if (!href) {
        return;
      }

      event.preventDefault();

      if (isLoading) {
        return;
      }

      isLoading = true;
      overlay.classList.add('is-visible');
      overlay.setAttribute('aria-hidden', 'false');

      const delay = prefersReducedMotion.matches ? 450 : 1700;
      window.setTimeout(function () {
        window.location.href = href;
      }, delay);
    });
  }

  function initParallax(prefersReducedMotion) {
    if (prefersReducedMotion.matches) {
      return;
    }

    const hero = document.querySelector('.hero');
    const visual = document.querySelector('.hero__visual');

    if (!hero || !visual) {
      return;
    }

    let animationFrameId = null;

    const reset = function () {
      visual.style.setProperty('--hero-translate-x', '0px');
      visual.style.setProperty('--hero-translate-y', '0px');
    };

    const handleMove = function (event) {
      if (animationFrameId) {
        window.cancelAnimationFrame(animationFrameId);
      }

      animationFrameId = window.requestAnimationFrame(function () {
        const rect = hero.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) - 0.5;
        const y = ((event.clientY - rect.top) / rect.height) - 0.5;
        const maxOffset = 14;

        visual.style.setProperty('--hero-translate-x', (x * maxOffset).toFixed(2) + 'px');
        visual.style.setProperty('--hero-translate-y', (y * maxOffset).toFixed(2) + 'px');
      });
    };

    hero.addEventListener('pointermove', handleMove);
    hero.addEventListener('pointerleave', reset);
    hero.addEventListener('pointercancel', reset);
  }
})();
