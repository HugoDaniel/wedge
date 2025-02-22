/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/io/progress.ts

*/

import { assert } from '../../../util';
import { OnProgressCallback } from './types';

/**
 * Monitor Promise.all progress, fire onProgress callback function.
 *
 * @param promises Promise list going to be monitored
 * @param onProgress Callback function. Fired when a promise resolved.
 * @param startFraction Optional fraction start. Default to 0.
 * @param endFraction Optional fraction end. Default to 1.
 */
export function monitorPromisesProgress<T>(
  promises: Array<Promise<T>>, onProgress: OnProgressCallback,
  startFraction?: number, endFraction?: number) {
  checkPromises(promises);
  startFraction = startFraction == null ? 0 : startFraction;
  endFraction = endFraction == null ? 1 : endFraction;
  checkFraction(startFraction, endFraction);
  let resolvedPromise = 0;

  const registerMonitor = (promise: Promise<T>) => {
    promise.then(value => {
      const fraction = startFraction +
        ++resolvedPromise / promises.length * (endFraction - startFraction);
      // pass fraction as parameter to callback function.
      onProgress(fraction);
      return value;
    });
    return promise;
  };

  function checkPromises(promises: Array<Promise<T>>): void {
    assert(
      promises != null && Array.isArray(promises) && promises.length > 0,
      () => 'promises must be a none empty array');
  }

  function checkFraction(startFraction: number, endFraction: number): void {
    assert(
      startFraction >= 0 && startFraction <= 1,
      () => `Progress fraction must be in range [0, 1], but ` +
        `got startFraction ${startFraction}`);
    assert(
      endFraction >= 0 && endFraction <= 1,
      () => `Progress fraction must be in range [0, 1], but ` +
        `got endFraction ${endFraction}`);
    assert(
      endFraction >= startFraction,
      () => `startFraction must be no more than endFraction, but ` +
        `got startFraction ${startFraction} and endFraction ` +
        `${endFraction}`);
  }

  return Promise.all(promises.map(registerMonitor));
}