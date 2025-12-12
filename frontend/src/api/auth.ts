import api, { setTokens } from './client';
import type { Token, User, LoginRequest, RegisterRequest } from '../types';

export const authApi = {
  async login(data: LoginRequest): Promise<Token> {
    const response = await api.post<Token>('/auth/login', data);
    const { access_token, refresh_token } = response.data;
    setTokens(access_token, refresh_token);
    return response.data;
  },

  async register(data: RegisterRequest): Promise<User> {
    const response = await api.post<User>('/auth/register', data);
    return response.data;
  },

  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>('/users/me');
    return response.data;
  },

  async updateUser(data: Partial<User>): Promise<User> {
    const response = await api.patch<User>('/users/me', data);
    return response.data;
  },
};
