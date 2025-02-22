/*

Original:
https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/io/router_registry.ts

*/

import { IOHandler, LoadOptions } from './types';

export type IORouter = (url: string | string[], loadOptions?: LoadOptions) =>
  IOHandler | null;

export class IORouterRegistry {
  // Singleton instance.
  private static instance: IORouterRegistry;

  private saveRouters: IORouter[];
  private loadRouters: IORouter[];

  private constructor() {
    this.saveRouters = [];
    this.loadRouters = [];
  }

  private static getInstance(): IORouterRegistry {
    if (IORouterRegistry.instance == null) {
      IORouterRegistry.instance = new IORouterRegistry();
    }
    return IORouterRegistry.instance;
  }

  /**
   * Register a save-handler router.
   *
   * @param saveRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `save` method defined or `null`.
   */
  static registerSaveRouter(saveRouter: IORouter) {
    IORouterRegistry.getInstance().saveRouters.push(saveRouter);
  }

  /**
   * Register a load-handler router.
   *
   * @param loadRouter A function that maps a URL-like string onto an instance
   * of `IOHandler` with the `load` method defined or `null`.
   */
  static registerLoadRouter(loadRouter: IORouter) {
    IORouterRegistry.getInstance().loadRouters.push(loadRouter);
  }

  /**
   * Look up IOHandler for saving, given a URL-like string.
   *
   * @param url
   * @returns If only one match is found, an instance of IOHandler with the
   * `save` method defined. If no match is found, `null`.
   * @throws Error, if more than one match is found.
   */
  static getSaveHandlers(url: string | string[]): IOHandler[] {
    return IORouterRegistry.getHandlers(url, 'save');
  }

  /**
   * Look up IOHandler for loading, given a URL-like string.
   *
   * @param url
   * @param loadOptions Optional, custom load options.
   * @returns All valid handlers for `url`, given the currently registered
   *   handler routers.
   */
  static getLoadHandlers(url: string | string[], loadOptions?: LoadOptions):
    IOHandler[] {
    return IORouterRegistry.getHandlers(url, 'load', loadOptions);
  }

  private static getHandlers(
    url: string | string[], handlerType: 'save' | 'load',
    loadOptions?: LoadOptions): IOHandler[] {
    const validHandlers: IOHandler[] = [];
    const routers = handlerType === 'load' ?
      IORouterRegistry.getInstance().loadRouters :
      IORouterRegistry.getInstance().saveRouters;
    routers.forEach(router => {
      const handler = router(url, loadOptions);
      if (handler !== null) {
        validHandlers.push(handler);
      }
    });
    return validHandlers;
  }
}

export const registerSaveRouter = (loudRouter: IORouter) =>
  IORouterRegistry.registerSaveRouter(loudRouter);
export const registerLoadRouter = (loudRouter: IORouter) =>
  IORouterRegistry.registerLoadRouter(loudRouter);
export const getSaveHandlers = (url: string | string[]) =>
  IORouterRegistry.getSaveHandlers(url);
export const getLoadHandlers =
  (url: string | string[], loadOptions?: LoadOptions) =>
    IORouterRegistry.getLoadHandlers(url, loadOptions);