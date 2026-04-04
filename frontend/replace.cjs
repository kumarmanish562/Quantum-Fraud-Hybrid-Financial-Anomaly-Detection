const fs = require('fs');
const path = require('path');

function processFile(filePath) {
  let content = fs.readFileSync(filePath, 'utf8');

  // Background and border replacements
  content = content.replace(/bg-\[\#111827\]/g, 'bg-white');
  content = content.replace(/bg-\[\#0B0F1A\]/g, 'bg-gray-50');
  content = content.replace(/bg-gray-800\/50/g, 'bg-gray-100');
  content = content.replace(/bg-gray-800\/20/g, 'bg-gray-50');
  content = content.replace(/bg-gray-800/g, 'bg-white');
  content = content.replace(/bg-gray-900\/30/g, 'bg-gray-100');
  content = content.replace(/bg-gray-900/g, 'bg-gray-50');
  content = content.replace(/border-gray-800\/50/g, 'border-gray-200');
  content = content.replace(/border-gray-800/g, 'border-gray-200');
  content = content.replace(/border-gray-700\/50/g, 'border-gray-300');
  content = content.replace(/border-gray-700/g, 'border-gray-300');
  
  // Text color replacements
  content = content.replace(/text-gray-400/g, 'text-gray-500');
  content = content.replace(/text-gray-300/g, 'text-gray-600');
  
  // Replace text-white safely: only inside classNames, BUT not when bg-(blue|green|red|gradient) is in the same line
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes('text-white')) {
      const skipKeywords = ['bg-blue-', 'bg-green-', 'bg-red-', 'bg-gradient', 'from-blue', 'text-white text-xs rounded-full', 'bg-cyan', 'bg-purple'];
      let shouldSkip = false;
      for (const kw of skipKeywords) {
        if (lines[i].includes(kw)) {
          shouldSkip = true;
          break;
        }
      }
      
      // Some text-white are in the same block as a colored background badge
      if (!shouldSkip) {
        lines[i] = lines[i].replace(/text-white/g, 'text-gray-900');
      }
    }
  }
  content = lines.join('\n');

  fs.writeFileSync(filePath, content);
}

function traverse(dir) {
  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory() && file !== 'node_modules' && file !== 'dist') {
      traverse(fullPath);
    } else if (fullPath.endsWith('.jsx') || fullPath.endsWith('.js') || fullPath.endsWith('.html')) {
      processFile(fullPath);
    }
  }
}

traverse(path.join(__dirname, 'src'));
console.log('Theme changed successfully.');
