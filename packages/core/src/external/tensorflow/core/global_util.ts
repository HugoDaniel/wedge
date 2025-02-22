/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/global_util.ts

*/

let globalNameSpace: { _tfGlobals: Map<string, any> };
// tslint:disable-next-line:no-any
export function getGlobalNamespace(): { _tfGlobals: Map<string, any> } {
  if (globalNameSpace == null) {
    // tslint:disable-next-line:no-any
    let ns: any;
    if (typeof (window) !== 'undefined') {
      ns = window;
    } else if (typeof (global) !== 'undefined') {
      ns = global;
    } else if (typeof (process) !== 'undefined') {
      ns = process;
    } else if (typeof (self) !== 'undefined') {
      ns = self;
    } else {
      throw new Error('Could not find a global object');
    }
    globalNameSpace = ns;
  }
  return globalNameSpace;
}

// tslint:disable-next-line:no-any
function getGlobalMap(): Map<string, any> {
  const ns = getGlobalNamespace();
  if (ns._tfGlobals == null) {
    ns._tfGlobals = new Map();
  }
  return ns._tfGlobals;
}

/**
 * Returns a globally accessible 'singleton' object.
 *
 * @param key the name of the object
 * @param init a function to initialize to initialize this object
 *             the first time it is fetched.
 */
export function getGlobal<T>(key: string, init: () => T): T {
  const globalMap = getGlobalMap();
  if (globalMap.has(key)) {
    return globalMap.get(key);
  } else {
    const singleton = init();
    globalMap.set(key, singleton);
    return globalMap.get(key);
  }
}